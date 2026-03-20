import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from PIL import Image
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from torchvision import transforms

from app.core.config import settings

logger = logging.getLogger(__name__)

ARTIFACTS = Path(settings.artifacts_root)

DL_MODEL_PATHS: Dict[str, Path] = {
    "resnet50": ARTIFACTS / "training_dl" / "resnet50_best_model.pth",
    "mobilenet_v2": ARTIFACTS / "training_dl" / "mobilenet_v2_best_model.pth",
    "efficientnet_b0": ARTIFACTS / "training_dl" / "efficientnet_b0_best_model.pth",
}
ML_MODEL_PATH = ARTIFACTS / "training_ml" / "xgboost_model.json"
DATASET_DIR = ARTIFACTS / "data_ingestion" / "images" / "images"

# Hardcoded fallback — sorted class names from the waste dataset
_FALLBACK_CLASSES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "clothing", "coffee_grounds", "disposable_plastic_cutlery",
    "eggshells", "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups",
    "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags",
    "plastic_water_bottles", "shoes", "steel_food_cans", "styrofoam_cups",
    "styrofoam_food_containers", "tea_bags",
]


def _discover_class_names() -> List[str]:
    """Read class names from the dataset directory, falling back to hardcoded list."""
    if DATASET_DIR.exists():
        names = sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir())
        if names:
            return names
    return _FALLBACK_CLASSES


def _extract_xgb_features(image: Image.Image) -> np.ndarray:
    """
    Extract 500-D local maxima features — identical to FeatureEngineeringMaster
    used during training.  Each peak contributes (x, y) → 250 peaks × 2 = 500 features.
    """
    img_arr = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    if img_arr.ndim == 3:
        gray = rgb2gray(img_arr)
    else:
        gray = img_arr
    gray = img_as_float(gray)
    coords = peak_local_max(gray, min_distance=10, num_peaks=250)
    features = np.zeros(500, dtype=np.float32)
    for i, (y_c, x_c) in enumerate(coords[:250]):
        features[2 * i] = x_c
        features[2 * i + 1] = y_c
    return features


class Predictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._class_names: Optional[List[str]] = None
        self._dl_cache: Dict[str, nn.Module] = {}
        self._xgb_cache: Optional[xgb.Booster] = None

    @property
    def class_names(self) -> List[str]:
        if self._class_names is None:
            self._class_names = _discover_class_names()
        return self._class_names

    def _load_dl_model(self, model_name: str) -> nn.Module:
        if model_name in self._dl_cache:
            return self._dl_cache[model_name]
        path = DL_MODEL_PATHS.get(model_name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"DL model '{model_name}' not found at {path}")
        model = torch.load(path, map_location=self.device, weights_only=False)
        model.eval()
        self._dl_cache[model_name] = model
        logger.info(f"Loaded DL model '{model_name}' from {path}")
        return model

    def _load_xgb_model(self) -> xgb.Booster:
        if self._xgb_cache is not None:
            return self._xgb_cache
        if not ML_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found at {ML_MODEL_PATH}")
        booster = xgb.Booster()
        booster.load_model(str(ML_MODEL_PATH))
        self._xgb_cache = booster
        logger.info(f"Loaded XGBoost model from {ML_MODEL_PATH}")
        return booster

    def predict_dl(self, run_id: str, image: Image.Image) -> Tuple[str, float]:
        """run_id is the model name: resnet50 | mobilenet_v2 | efficientnet_b0."""
        model = self._load_dl_model(run_id)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        idx = int(pred_idx.item())
        names = self.class_names
        class_name = names[idx] if idx < len(names) else str(idx)
        return class_name, float(confidence.item())

    def predict_ml(self, run_id: str, image: Image.Image) -> Tuple[str, float]:
        """Predict with XGBoost using local maxima feature engineering."""
        booster = self._load_xgb_model()
        features = _extract_xgb_features(image)
        dtest = xgb.DMatrix(features.reshape(1, -1))
        preds = booster.predict(dtest)
        # multi:softmax returns the predicted class index as a float
        pred_idx = int(round(float(preds[0])))
        names = self.class_names
        class_name = names[pred_idx] if pred_idx < len(names) else str(pred_idx)
        return class_name, 1.0
