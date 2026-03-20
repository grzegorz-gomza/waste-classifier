import io
import mimetypes
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import settings

router = APIRouter()

ARTIFACTS = Path(settings.artifacts_root)


def _run_dir(run_id: str) -> Path:
    """Return the reports directory for a given run_id (= model name)."""
    return ARTIFACTS / "reports" / run_id


@router.get("/{run_id}/list")
def list_artifacts(run_id: str):
    """List artifact files for a given run (reports directory)."""
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        return []
    result = []
    for f in run_dir.rglob("*"):
        result.append(
            {
                "path": str(f.relative_to(run_dir)),
                "is_dir": f.is_dir(),
                "size": f.stat().st_size if f.is_file() else None,
            }
        )
    return result


@router.get("/{run_id}/download")
def download_artifact(run_id: str, path: str = ""):
    """Download a specific artifact file or zip a directory."""
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"No artifacts for run '{run_id}'")

    target = run_dir / path if path else run_dir

    if target.is_file():
        media_type, _ = mimetypes.guess_type(str(target))
        return FileResponse(
            target,
            media_type=media_type or "application/octet-stream",
            filename=target.name,
        )

    if target.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in target.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(target))
        buf.seek(0)
        name = target.name or "artifacts"
        return StreamingResponse(
            io.BytesIO(buf.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={name}.zip"},
        )

    raise HTTPException(status_code=404, detail=f"Path '{path}' not found in run '{run_id}'")
