from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision import transforms
from tqdm import tqdm

from WasteClassifier.components.machine_learning.feature_engineering import FeatureEngineeringMaster
from WasteClassifier.components.share.dataset import WasteDataset
from WasteClassifier.components.visualization.plot_artifacts import PlotArtifacts, PlotConfig
from WasteClassifier.entity.config_entity import EvaluationConfig, TrackingConfig, VisualizationConfig
from WasteClassifier.utils.mlflow_utils import (
    log_artifacts_dir,
    log_metrics,
    read_run_context,
    resume_run,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvalResult:
    run_id: str
    accuracy: float
    precision: float
    recall: float
    f1_weighted: float
    confusion_matrix: np.ndarray


class EvaluateModels:
    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        tracking_config: TrackingConfig,
        visualization_config: VisualizationConfig,
    ):
        self.evaluation_config = evaluation_config
        self.tracking_config = tracking_config
        self.visualization_config = visualization_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_test_dataset(self) -> WasteDataset:
        image_size = tuple(self.evaluation_config.params_image_size[:2])
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        return WasteDataset(
            root_dir=self.evaluation_config.test_data_path,
            transform=transform,
            split="test",
            test_split=0.2,
        )

    def _evaluate_dl_model(self, model_name: str, model_path: Path, test_dataset: WasteDataset) -> ModelEvalResult:
        model = torch.load(model_path, weights_only=False)
        model = model.to(self.device)
        model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for img, label in test_dataset:
                logits = model(img.unsqueeze(0).to(self.device))
                pred = int(torch.argmax(logits, dim=1).item())
                y_true.append(int(label))
                y_pred.append(pred)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)

        return ModelEvalResult(
            run_id=model_name,
            accuracy=accuracy,
            precision=float(precision),
            recall=float(recall),
            f1_weighted=float(f1_weighted),
            confusion_matrix=cm,
        )

    def _evaluate_xgboost(self, model_path: Path, test_dataset: WasteDataset) -> ModelEvalResult:
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        fe_master = FeatureEngineeringMaster()
        X_test: List[np.ndarray] = []
        y_true: List[int] = []

        for img, label in tqdm(test_dataset, desc="Feature engineering test set"):
            img_np = img.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            X_test.append(fe_master.extract_all_features(img_np))
            y_true.append(int(label))

        dtest = xgb.DMatrix(np.asarray(X_test), label=np.asarray(y_true))
        pred_raw = booster.predict(dtest)
        if len(pred_raw.shape) == 1:
            y_pred = pred_raw.astype(int)
        else:
            y_pred = np.argmax(pred_raw, axis=1)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1_weighted, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)

        return ModelEvalResult(
            run_id="xgboost",
            accuracy=accuracy,
            precision=float(precision),
            recall=float(recall),
            f1_weighted=float(f1_weighted),
            confusion_matrix=cm,
        )

    def _write_metrics(self, report_dir: Path, result: ModelEvalResult) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": result.run_id,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_weighted": result.f1_weighted,
            "f1_score": result.f1_weighted,
        }
        (report_dir / "evaluation_metrics.json").write_text(json.dumps(payload, indent=2))

    def _latest_context_for(self, model_type: str, model_name: str) -> Optional[Tuple[str, Path]]:
        if not self.tracking_config.runs_root_dir.exists():
            return None

        contexts = sorted(
            self.tracking_config.runs_root_dir.glob("*/run_context.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for ctx_path in contexts:
            try:
                ctx = read_run_context(ctx_path)
            except Exception:
                continue
            if str(ctx.get("model_type")) == model_type and str(ctx.get("model_name")) == model_name:
                run_id = str(ctx.get("run_id", ""))
                report_dir = Path(str(ctx.get("report_dir", Path("artifacts") / "reports" / model_name)))
                return run_id, report_dir
        return None

    def _resume_and_log(self, run_id: str, report_dir: Path, result: ModelEvalResult) -> None:
        if not self.tracking_config.enabled or not run_id:
            return

        resume_run(
            run_id=run_id,
            tracking_uri=self.tracking_config.tracking_uri,
            experiment_name=self.tracking_config.experiment_name,
        )
        log_metrics(
            {
                "eval_accuracy": result.accuracy,
                "eval_precision": result.precision,
                "eval_recall": result.recall,
                "eval_f1_weighted": result.f1_weighted,
            }
        )
        log_artifacts_dir(report_dir, artifact_path="reports")

        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass

    def _evaluate_single(
        self,
        *,
        run_id: str,
        model_type: str,
        model_name: str,
        model_path: Path,
        test_dataset: WasteDataset,
        class_names: List[str],
    ) -> Optional[ModelEvalResult]:
        if not model_path.exists():
            logger.warning("Model not found, skipping: %s", model_path)
            return None

        if model_type == "DL":
            result = self._evaluate_dl_model(model_name, model_path, test_dataset)
        else:
            result = self._evaluate_xgboost(model_path, test_dataset)

        report_dir = Path("artifacts") / "reports" / run_id
        self._write_metrics(report_dir, result)

        plotter = PlotArtifacts(
            report_dir,
            config=PlotConfig(enabled=self.visualization_config.enabled, dpi=self.visualization_config.dpi),
        )
        plotter.plot_confusion_matrix(
            run_name=run_id,
            cm=result.confusion_matrix,
            class_names=class_names,
            title=f"{run_id} - Confusion Matrix",
            file_suffix=model_type.lower(),
            output_name="confusion_matrix.png",
            normalize=False,
        )

        latest_context = self._latest_context_for(model_type, model_name)
        if latest_context is not None:
            tracked_run_id, _ = latest_context
            self._resume_and_log(tracked_run_id, report_dir, result)

        return result

    def main(self) -> None:
        test_dataset = self._get_test_dataset()
        if len(test_dataset) == 0:
            raise RuntimeError("No test samples found. Run data ingestion first.")

        class_names = test_dataset.get_class_names()
        all_results: Dict[str, Dict[str, float]] = {}

        dl_models = ["resnet50", "mobilenet_v2", "efficientnet_b0"]
        for model_name in dl_models:
            result = self._evaluate_single(
                run_id=model_name,
                model_type="DL",
                model_name=model_name,
                model_path=Path("artifacts") / "training_dl" / f"{model_name}_best_model.pth",
                test_dataset=test_dataset,
                class_names=class_names,
            )
            if result is not None:
                all_results[result.run_id] = {
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_weighted": result.f1_weighted,
                }

        xgb_result = self._evaluate_single(
            run_id="xgboost",
            model_type="ML",
            model_name="xgboost",
            model_path=Path("artifacts") / "training_ml" / "xgboost_model.json",
            test_dataset=test_dataset,
            class_names=class_names,
        )
        if xgb_result is not None:
            all_results[xgb_result.run_id] = {
                "accuracy": xgb_result.accuracy,
                "precision": xgb_result.precision,
                "recall": xgb_result.recall,
                "f1_weighted": xgb_result.f1_weighted,
            }

        metrics_dir = Path("artifacts") / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "combined_metrics.json").write_text(json.dumps(all_results, indent=2))

        logger.info("Unified evaluation completed")
