"""
Configuration entities for Waste Classification project.
All configuration dataclasses for different pipeline stages.

Author: Grzegorz Gomza
Date: February 2026
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: List[int]
    params_classes: int
    params_model_name: str
    params_pretrained: bool
    params_freeze_base: bool
    params_models: List[str]  # List of models to prepare

@dataclass(frozen=True)
class TrainDLModelConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_learning_rate: float
    params_image_size: List[int]
    params_augmentation: Dict[str, bool]  # Augmentation configuration
    params_test_split: float
    params_models: List[str]  # List of models to train
    params_optimizer: str  # Optimizer type
    params_weight_decay: float  # Weight decay for regularization
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    dl_model_path: Path
    test_data_path: Path
    dl_metrics_path: Path
    params_image_size: List[int]
    params_batch_size: int

@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tracking_uri: Optional[str]
    experiment_name: str
    runs_root_dir: Path

@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    dpi: int

@dataclass(frozen=True)
class TrainMLModelConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    params_image_size: List[int]
    params_test_split: float
    params_random_state: int
    params_xgb_num_boost_round: int
    params_xgb_max_depth: int
    params_xgb_eta: float
    params_xgb_subsample: float
    params_xgb_colsample_bytree: float
    params_xgb_objective: str
    params_xgb_eval_metric: str
    params_xgb_batch_size: int
