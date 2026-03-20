from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torchvision import transforms

from WasteClassifier.components.machine_learning.feature_engineering import FeatureEngineeringMaster
from WasteClassifier.components.share.dataset import WasteDataset
from WasteClassifier.components.visualization.plot_artifacts import PlotArtifacts, PlotConfig
from WasteClassifier.entity.config_entity import TrainMLModelConfig, TrackingConfig, VisualizationConfig
from WasteClassifier.utils.mlflow_utils import (
    log_artifacts_dir,
    log_metrics,
    log_params,
    make_run_name,
    start_run,
    write_run_context,
)

logger = logging.getLogger(__name__)


class TrainXGBModel:
    def __init__(
        self,
        config: TrainMLModelConfig,
        tracking_config: TrackingConfig,
        visualization_config: VisualizationConfig,
    ):
        self.config = config
        self.tracking_config = tracking_config
        self.visualization_config = visualization_config

    def train(self) -> None:
        logger.info("Starting XGBoost training with engineered features...")

        fe_master = FeatureEngineeringMaster()
        dataset = WasteDataset(
            root_dir=str(self.config.training_data),
            transform=transforms.Compose(
                [
                    transforms.Resize(self.config.params_image_size[:2]),
                    transforms.ToTensor(),
                ]
            ),
        )

        dataset_size = len(dataset)
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        train_data_fraction = float(self.config.params_xgb_train_data_fraction)
        limited_dataset_size = int(dataset_size * train_data_fraction)
        logger.info(
            "Using %.0f%% of dataset: %s/%s samples",
            train_data_fraction * 100,
            limited_dataset_size,
            dataset_size,
        )

        if train_data_fraction < 1.0:
            import random

            random.seed(42)
            all_indices = list(range(dataset_size))
            selected_indices = random.sample(all_indices, limited_dataset_size)
            dataset = torch.utils.data.Subset(dataset, selected_indices)

        cache_file = fe_master.cache_dir / f"feature_engineered_data_{train_data_fraction}.pkl"
        X = None
        y = None

        if cache_file.exists():
            logger.info("Loading cached feature-engineered data from: %s", cache_file)
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                X = cache_data["X"]
                y = cache_data["y"]
                logger.info("Loaded cached data: %s samples, %s features", X.shape[0], X.shape[1])
            except Exception as e:
                logger.warning("Failed to load cached data: %s", e)

        if X is None or y is None:
            logger.info("Extracting features for all images...")
            X_list = []
            y_list = []
            for img, label in tqdm(dataset, desc="Feature engineering all images"):
                img_np = img.numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                X_list.append(fe_master.extract_all_features(img_np))
                y_list.append(label)

            X = np.array(X_list)
            y = np.array(y_list)

            cache_data = {
                "X": X,
                "y": y,
                "feature_names": fe_master.get_feature_names(),
                "metadata": {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "extraction_time": datetime.now().isoformat(),
                    "dataset_fraction": train_data_fraction,
                },
            }
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info("Feature-engineered data cached at: %s", cache_file)
            except Exception as e:
                logger.warning("Failed to save cache data: %s", e)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=float(self.config.params_test_split),
            random_state=int(self.config.params_random_state),
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        logger.info("Train/val split: %s train, %s validation", X_train.shape[0], X_val.shape[0])

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": self.config.params_xgb_objective,
            "num_class": num_classes,
            "eval_metric": self.config.params_xgb_eval_metric,
            "max_depth": self.config.params_xgb_max_depth,
            "eta": self.config.params_xgb_eta,
            "subsample": self.config.params_xgb_subsample,
            "colsample_bytree": self.config.params_xgb_colsample_bytree,
            "seed": self.config.params_random_state,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        run_name = make_run_name("ML", "xgboost", datetime.now())
        report_dir = Path("artifacts") / "reports" / "xgboost"
        report_dir.mkdir(parents=True, exist_ok=True)

        active_run = None
        if self.tracking_config.enabled:
            active_run = start_run(
                run_name=run_name,
                tracking_uri=self.tracking_config.tracking_uri,
                experiment_name=self.tracking_config.experiment_name,
                tags={
                    "model_type": "ML",
                    "model_name": "xgboost",
                    "run_name": run_name,
                },
            )
            log_params(
                {
                    "model_type": "ML",
                    "model_name": "xgboost",
                    "train_samples": int(X_train.shape[0]),
                    "val_samples": int(X_val.shape[0]),
                    "num_features": int(X_train.shape[1]),
                    "num_classes": int(num_classes),
                    "train_data_fraction": float(train_data_fraction),
                    "feature_engineering": "peak_coordinates_500",
                    "max_depth": int(self.config.params_xgb_max_depth),
                    "eta": float(self.config.params_xgb_eta),
                    "subsample": float(self.config.params_xgb_subsample),
                    "colsample_bytree": float(self.config.params_xgb_colsample_bytree),
                    "objective": self.config.params_xgb_objective,
                    "eval_metric": self.config.params_xgb_eval_metric,
                    "device": params["device"],
                }
            )

        evals_result: dict = {}
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.params_xgb_num_boost_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
            early_stopping_rounds=10,
        )

        val_preds_raw = bst.predict(dval)
        if len(val_preds_raw.shape) == 1:
            val_preds = val_preds_raw.astype(int)
        else:
            val_preds = np.argmax(val_preds_raw, axis=1)

        val_acc = float(accuracy_score(y_val, val_preds))
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val,
            val_preds,
            average="weighted",
            zero_division=0,
        )

        logger.info("Final validation accuracy: %.4f", val_acc)
        bst.save_model(str(self.config.trained_model_path))
        logger.info("Model saved to %s", self.config.trained_model_path)

        training_curves_path = report_dir / "training_curves.json"
        training_curves_path.write_text(json.dumps(evals_result, indent=2))

        plotter = PlotArtifacts(
            report_dir,
            config=PlotConfig(enabled=self.visualization_config.enabled, dpi=self.visualization_config.dpi),
        )
        plotter.plot_ml_training_progress(
            run_name="xgboost",
            evals_result=evals_result,
            metric_key="mlogloss",
            output_name="training_progress.png",
        )

        if active_run is not None:
            log_metrics(
                {
                    "val_accuracy": val_acc,
                    "val_precision": float(val_precision),
                    "val_recall": float(val_recall),
                    "val_f1_weighted": float(val_f1),
                }
            )
            log_artifacts_dir(report_dir, artifact_path="reports")
            write_run_context(
                context_path=self.tracking_config.runs_root_dir / run_name / "run_context.json",
                run_name=run_name,
                run_id=active_run.info.run_id,
                model_type="ML",
                model_name="xgboost",
                report_dir=report_dir,
            )
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass

        logger.info("XGBoost training completed successfully")

    def main(self) -> None:
        self.train()
