from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile

from src.inference import predict as run_prediction
from src.guardrails import apply_safety_guardrails
from src.database import insert_run

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.1.0")
UPLOAD_DIR = Path("tmp_uploads")
DB_PATH = Path(__file__).resolve().parent.parent / "medical_ai_evidence.sqlite"


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    pred = apply_safety_guardrails(run_prediction(target, mode="improved_v3"))
    
    # Save the run to the database
    case_id = f"api_{safe_stem}_{int(time.time())}"
    insert_run(DB_PATH, case_id, str(target), pred, dataset_source="api_upload")
    
    return pred

