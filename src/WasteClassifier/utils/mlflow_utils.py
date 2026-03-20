import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def make_run_name(model_type: str, model_name: str, timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    return f"{model_type}_{model_name}_{ts.strftime('%Y_%m_%d_%H_%M')}"


def safe_end_active_run() -> None:
    run = mlflow.active_run()
    if run is not None:
        try:
            mlflow.end_run()
        except Exception:
            # If mlflow already ended the run or backend rejects, ignore to match example-style robustness.
            pass


def configure_mlflow(tracking_uri: Optional[str], experiment_name: Optional[str]) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)


def start_run(
    *,
    run_name: str,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.ActiveRun:
    safe_end_active_run()
    configure_mlflow(tracking_uri, experiment_name)

    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run


def resume_run(
    *,
    run_id: str,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.ActiveRun:
    safe_end_active_run()
    configure_mlflow(tracking_uri, experiment_name)

    run = mlflow.start_run(run_id=run_id)
    if tags:
        mlflow.set_tags(tags)
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    cleaned: Dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            cleaned[k] = float(v)
        except Exception:
            continue

    if cleaned:
        mlflow.log_metrics(cleaned, step=step)


def log_params(params: Dict[str, Any]) -> None:
    cleaned: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        # Ensure JSON-serializable, otherwise string it.
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)

    if cleaned:
        mlflow.log_params(cleaned)


def log_artifacts_dir(local_dir: Path, artifact_path: Optional[str] = None) -> None:
    if local_dir.exists():
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)


def write_run_context(
    *,
    context_path: Path,
    run_name: str,
    run_id: str,
    model_type: str,
    model_name: str,
    report_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "model_type": model_type,
        "model_name": model_name,
        "report_dir": str(report_dir),
    }
    if extra:
        payload.update(extra)

    context_path.write_text(json.dumps(payload, indent=2))


def read_run_context(context_path: Path) -> Dict[str, Any]:
    return json.loads(context_path.read_text())
