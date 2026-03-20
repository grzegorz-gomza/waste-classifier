import json
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter()

ARTIFACTS = Path(settings.artifacts_root)

# Static model registry — mirrors frontend models.ts
# run_id doubles as the model identifier used by the Predictor
KNOWN_MODELS: List[Dict[str, str]] = [
    {
        "run_id": "resnet50",
        "run_name": "CNN Model A (ResNet50)",
        "model_type": "DL",
        "model_name": "resnet50",
        "artifact_path": "training_dl/resnet50_best_model.pth",
    },
    {
        "run_id": "mobilenet_v2",
        "run_name": "CNN Model B (MobileNetV2)",
        "model_type": "DL",
        "model_name": "mobilenet_v2",
        "artifact_path": "training_dl/mobilenet_v2_best_model.pth",
    },
    {
        "run_id": "efficientnet_b0",
        "run_name": "CNN Model C (EfficientNetB0)",
        "model_type": "DL",
        "model_name": "efficientnet_b0",
        "artifact_path": "training_dl/efficientnet_b0_best_model.pth",
    },
    {
        "run_id": "xgboost",
        "run_name": "Random Forest Model (XGBoost)",
        "model_type": "ML",
        "model_name": "xgboost",
        "artifact_path": "training_ml/xgboost_model.json",
    },
]


def _load_metrics(run_id: str) -> Dict[str, float]:
    """Load evaluation metrics from disk for a given run_id (= model name)."""
    candidates = [
        ARTIFACTS / "reports" / run_id / "evaluation_metrics.json",
        ARTIFACTS / "evaluation" / f"{run_id}_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                metrics: Dict[str, float] = {}
                for key in ("accuracy", "f1_weighted", "f1_score", "precision", "recall"):
                    if key in data and isinstance(data[key], (int, float)):
                        metrics[key] = float(data[key])
                # Expose f1_weighted as f1_score so the evaluation page picks it up
                if "f1_weighted" in metrics and "f1_score" not in metrics:
                    metrics["f1_score"] = metrics["f1_weighted"]
                return metrics
            except Exception:
                pass
    return {}


def _load_plot_urls(run_id: str) -> Dict[str, str]:
    """Resolve deterministic plot artifact URLs for a given run/model."""
    run_dir = ARTIFACTS / "reports" / run_id
    candidates = {
        "training_progress": "training_progress.png",
        "confusion_matrix": "confusion_matrix.png",
    }
    plots: Dict[str, str] = {}
    for key, filename in candidates.items():
        path = run_dir / filename
        if path.exists():
            plots[key] = f"/api/artifacts/{run_id}/download?path={filename}"
    return plots


@router.get("/", response_model=List[Dict[str, Any]])
def list_runs():
    """Return all models whose artifact files exist on disk as virtual runs."""
    now_ms = int(time.time() * 1000)
    result = []
    for m in KNOWN_MODELS:
        model_file = ARTIFACTS / m["artifact_path"]
        if not model_file.exists():
            continue
        result.append(
            {
                "run_id": m["run_id"],
                "run_name": m["run_name"],
                "model_type": m["model_type"],
                "model_name": m["model_name"],
                "status": "FINISHED",
                "start_time": now_ms,
                "end_time": now_ms,
                "artifact_uri": str(model_file),
            }
        )
    return result


@router.get("/{run_id}", response_model=Dict[str, Any])
def get_run(run_id: str):
    """Return run details + evaluation metrics for a given model (by run_id)."""
    model = next((m for m in KNOWN_MODELS if m["run_id"] == run_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    model_file = ARTIFACTS / model["artifact_path"]
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found for run '{run_id}'")

    now_ms = int(time.time() * 1000)
    metrics = _load_metrics(run_id)
    plots = _load_plot_urls(run_id)

    return {
        "run_id": run_id,
        "run_name": model["run_name"],
        "model_type": model["model_type"],
        "model_name": model["model_name"],
        "status": "FINISHED",
        "start_time": now_ms,
        "end_time": now_ms,
        "artifact_uri": str(model_file),
        "params": {},
        "metrics": metrics,
        "plots": plots,
        "tags": {
            "model_type": model["model_type"],
            "model_name": model["model_name"],
        },
    }
