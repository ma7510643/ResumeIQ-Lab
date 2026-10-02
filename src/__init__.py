from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "resumeiq.db"

UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(UPLOAD_DIR / "resumes").mkdir(exist_ok=True)
(UPLOAD_DIR / "company").mkdir(exist_ok=True)
