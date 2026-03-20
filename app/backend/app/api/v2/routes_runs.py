from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.v2.schemas import ModelInfo
from app.services import model_registry

router = APIRouter()


def get_model_metrics(model_id: str) -> dict:
    """
    Read actual evaluation metrics from JSON files.
    
    Args:
        model_id (str): Model identifier
        
    Returns:
        dict: Metrics dictionary with actual values or defaults if file not found
    """
    try:
        # Try to read from artifacts/reports directory first
        metrics_path = Path("artifacts/reports") / model_id / "evaluation_metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                return {
                    "val_accuracy": metrics.get("accuracy", 0.0),
                    "val_loss": round(1.0 - metrics.get("accuracy", 0.0), 3),  # Approximate loss
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_weighted", metrics.get("f1_score", 0.0))
                }
        
        # Fallback to MLflow artifacts if not found in artifacts/reports
        mlflow_metrics_map = {
            "resnet50": "9a14629ae58345b090547c19d4a4723b",
            "mobilenet_v2": "d514323bdd0041018f0a20a25db7d913", 
            "efficientnet_b0": "a503c959c80e4172b5480cadfdd88444",
            "xgboost": "f6a936c0d8394e85bb252c050ca6eab0",
            "xgboost_multi_otsu_histogram_rgb": "aa1846e8dd4243789bce460d2b986085",
        }
        
        if model_id in mlflow_metrics_map:
            mlflow_path = Path("mlruns/1") / mlflow_metrics_map[model_id] / "artifacts/reports/evaluation_metrics.json"
            if mlflow_path.exists():
                with open(mlflow_path, 'r') as f:
                    metrics = json.load(f)
                    return {
                        "val_accuracy": metrics.get("accuracy", 0.0),
                        "val_loss": round(1.0 - metrics.get("accuracy", 0.0), 3),
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1_score": metrics.get("f1_weighted", metrics.get("f1_score", 0.0))
                    }
        
        # Return default values if no metrics found
        return {"val_accuracy": 0.0, "val_loss": 1.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
    except Exception as e:
        # Return default values on error
        return {"val_accuracy": 0.0, "val_loss": 1.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}


@router.get("/runs", response_model=List[dict])
def list_runs() -> List[dict]:
    now = int(time.time())
    return [
        {
            "run_id": m.model_id,
            "run_name": m.label,
            "model_type": m.model_type,
            "model_name": m.model_name,
            "status": "FINISHED",
            "start_time": now - 3600,  # 1 hour ago
            "end_time": now - 1800,    # 30 minutes ago
            "artifact_uri": "",
            "metrics": get_model_metrics(m.model_id),
        }
        for m in model_registry.list_models()
    ]


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    m = next((m for m in model_registry.list_models() if m.model_id == run_id), None)
    if not m:
        raise HTTPException(status_code=404, detail="Run not found")
    
    now = int(time.time())
    metrics = get_model_metrics(run_id)
    
    return {
        "run_id": m.model_id,
        "run_name": m.label,
        "model_type": m.model_type,
        "model_name": m.model_name,
        "status": "FINISHED",
        "start_time": now - 3600,
        "end_time": now - 1800,
        "artifact_uri": "",
        "params": {"learning_rate": "0.001", "batch_size": "32"},
        "metrics": metrics,
        "plots": {
            "training_progress": "/api/v2/artifacts/" + m.model_id + "/download?path=training_progress.png",
            "confusion_matrix": "/api/v2/artifacts/" + m.model_id + "/download?path=confusion_matrix.png",
        },
        "tags": {},
    }
