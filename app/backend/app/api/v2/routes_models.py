from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.api.v2.schemas import ModelInfo
from app.services import model_registry

router = APIRouter()


@router.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    # Return both v2 model_ids and legacy IDs for frontend compatibility
    base = settings.artifacts_root.rstrip("/")
    base_models = [
        ModelInfo(
            model_id="resnet50",
            model_type="DL",
            model_name="resnet50",
            artifact_path=f"{base}/training_dl/resnet50_best_model.pth",
            label="CNN Model A (ResNet50)",
        ),
        ModelInfo(
            model_id="mobilenet_v2",
            model_type="DL",
            model_name="mobilenet_v2",
            artifact_path=f"{base}/training_dl/mobilenet_v2_best_model.pth",
            label="CNN Model B (MobileNetV2)",
        ),
        ModelInfo(
            model_id="efficientnet_b0",
            model_type="DL",
            model_name="efficientnet_b0",
            artifact_path=f"{base}/training_dl/efficientnet_b0_best_model.pth",
            label="CNN Model C (EfficientNetB0)",
        ),
        ModelInfo(
            model_id="xgboost",
            model_type="ML",
            model_name="xgboost",
            artifact_path=f"{base}/training_ml/xgboost_model.json",
            label="XGBoost (Peak Local Max)",
        ),
        ModelInfo(
            model_id="xgboost_multi_otsu_histogram_rgb",
            model_type="ML",
            model_name="xgboost_multi_otsu_histogram_rgb",
            artifact_path=f"{base}/training_ml/xgboost_model_multi_otsu_histogram_rgb.json",
            label="XGBoost (Multi-Otsu RGB)",
        ),
    ]
    # Add legacy IDs for frontend compatibility
    legacy_entries = [
        ModelInfo(
            model_id="cnn-model-a",
            model_type="DL",
            model_name="resnet50",
            artifact_path=f"{base}/training_dl/resnet50_best_model.pth",
            label="CNN Model A (ResNet50)",
        ),
        ModelInfo(
            model_id="cnn-model-b",
            model_type="DL",
            model_name="mobilenet_v2",
            artifact_path=f"{base}/training_dl/mobilenet_v2_best_model.pth",
            label="CNN Model B (MobileNetV2)",
        ),
        ModelInfo(
            model_id="cnn-model-c",
            model_type="DL",
            model_name="efficientnet_b0",
            artifact_path=f"{base}/training_dl/efficientnet_b0_best_model.pth",
            label="CNN Model C (EfficientNetB0)",
        ),
        ModelInfo(
            model_id="random-forest-model",
            model_type="ML",
            model_name="xgboost",
            artifact_path=f"{base}/training_ml/xgboost_model.json",
            label="XGBoost (Peak Local Max)",
        ),
        ModelInfo(
            model_id="xgboost-multi-otsu-rgb",
            model_type="ML",
            model_name="xgboost_multi_otsu_histogram_rgb",
            artifact_path=f"{base}/training_ml/xgboost_model_multi_otsu_histogram_rgb.json",
            label="XGBoost (Multi-Otsu RGB)",
        ),
    ]
    return base_models + legacy_entries
