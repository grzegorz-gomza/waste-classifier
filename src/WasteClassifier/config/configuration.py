"""
Configuration manager for all pipeline stages.

Author: Grzegorz Gomza
Date: February 2026
References:
- Configuration pattern: https://github.com/entbappy/MLOps-Projects
"""

from pathlib import Path
from WasteClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from WasteClassifier.utils.common import read_yaml, create_directories
from WasteClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainDLModelConfig,
    EvaluationConfig,
    TrackingConfig,
    VisualizationConfig,
    TrainMLModelConfig,
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_filepath (Path): Path to config.yaml
            params_filepath (Path): Path to params.yaml
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        artifacts_root = getattr(self.config, "artifacts_root", "artifacts")
        create_directories([artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=getattr(config, "source_url", getattr(config, "source_URL")),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Get prepare base model configuration."""
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES,
            params_model_name=self.params.DL_MODELS[0] if self.params.DL_MODELS else 'resnet50',  
            params_pretrained=self.params.PRETRAINED,
            params_freeze_base=self.params.FREEZE_BASE,
            params_models=self.params.DL_MODELS
        )
    
    def get_train_dl_model_config(self) -> TrainDLModelConfig:
        """Get DL training configuration."""
        config = self.config.train_dl_model
        prepare_base_model = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        return TrainDLModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_image_size=self.params.IMAGE_SIZE,
            params_augmentation=self.params.AUGMENTATION,
            params_test_split=self.params.TEST_SPLIT,
            params_models=self.params.DL_MODELS,
            params_optimizer=self.params.OPTIMIZER,
            params_weight_decay=self.params.WEIGHT_DECAY
        )
        
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        config = self.config.evaluation
        
        create_directories([config.root_dir])
        
        return EvaluationConfig(
            root_dir=Path(config.root_dir),
            dl_model_path=Path(config.dl_model_path),
            test_data_path=Path(getattr(config, "test_data_path", self.config.data_ingestion.unzip_dir)),
            dl_metrics_path=Path(getattr(config, "dl_metrics_path", Path(config.root_dir) / "dl_metrics.json")),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

    def get_tracking_config(self) -> TrackingConfig:
        config = self.config.tracking
        create_directories([config.runs_root_dir])

        return TrackingConfig(
            enabled=bool(config.enabled),
            tracking_uri=None if config.tracking_uri in (None, "null", "None", "") else str(config.tracking_uri),
            experiment_name=str(config.experiment_name),
            runs_root_dir=Path(config.runs_root_dir),
        )

    def get_visualization_config(self) -> VisualizationConfig:
        config = self.config.visualization
        return VisualizationConfig(
            enabled=bool(config.enabled),
            dpi=int(config.dpi),
        )

    def get_train_ml_model_config(self) -> TrainMLModelConfig:
        config = self.config.train_ml_model
        create_directories([config.root_dir])

        return TrainMLModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            params_image_size=self.params.IMAGE_SIZE,
            params_test_split=self.params.TEST_SPLIT,
            params_random_state=int(self.params.XGB_RANDOM_STATE),
            params_xgb_num_boost_round=int(self.params.XGB_NUM_BOOST_ROUND),
            params_xgb_max_depth=int(self.params.XGB_MAX_DEPTH),
            params_xgb_eta=float(self.params.XGB_ETA),
            params_xgb_subsample=float(self.params.XGB_SUBSAMPLE),
            params_xgb_colsample_bytree=float(self.params.XGB_COLSAMPLE_BYTREE),
            params_xgb_objective=str(self.params.XGB_OBJECTIVE),
            params_xgb_eval_metric=str(self.params.XGB_EVAL_METRIC),
            params_xgb_batch_size=int(self.params.XGB_BATCH_SIZE),
        )
