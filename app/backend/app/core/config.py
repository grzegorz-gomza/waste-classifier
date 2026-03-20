from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve repo root relative to this file's location:
# app/backend/app/core/config.py -> 5 parents up -> 11_Assessment/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_ARTIFACTS = str(_REPO_ROOT / "artifacts")


class Settings(BaseSettings):
    # Path to the artifacts directory (override via ARTIFACTS_ROOT env var for Docker)
    artifacts_root: str = _DEFAULT_ARTIFACTS

    class Config:
        env_file = ".env"


settings = Settings()
