# app/main.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# Robust path handling (works locally + Render)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]     # repo root if app/ is inside repo
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model_pipeline.joblib"
MODEL_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
FRIENDLY_SCHEMA_PATH = ARTIFACT_DIR / "friendly_schema.json"


# --------------------------------------------------
# Load artifacts safely
# --------------------------------------------------
def _require(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"{name} not found at: {path.resolve()}")

_require(MODEL_PATH, "Model")
_require(MODEL_CONFIG_PATH, "Model config")
_require(FRIENDLY_SCHEMA_PATH, "Friendly schema")

model = joblib.load(MODEL_PATH)
model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
friendly_schema = json.loads(FRIENDLY_SCHEMA_PATH.read_text(encoding="utf-8"))

THRESHOLD = float(model_config.get("chosen_threshold", 0.45))
MODEL_FEATURES = model_config["all_features"]  # what the pipeline expects
FRIENDLY_FEATURES = friendly_schema["friendly_all_features"]  # what the API accepts


# You said appointment_id must exist for dashboard/table use
# It should NOT be part of MODEL_FEATURES unless you trained with it (you shouldn't).
ID_COL = "appointment_id"

NUMERIC_FRIENDLY = ["lead_time_days", "age", "appt_hour", "prev_no_shows", "hist_no_show_rate"]
CATEGORICAL_FRIENDLY = ["specialty", "clinic_name", "day_of_week", "insurance_type"]


# --------------------------------------------------
# FastAPI app + CORS (Streamlit ↔ API)
# --------------------------------------------------
app = FastAPI(title="HealthPulse No-Show Predictor", version="1.0")

# Allow local Streamlit + Render frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo; tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": THRESHOLD,
        "friendly_features": FRIENDLY_FEATURES,
        "model_features_count": len(MODEL_FEATURES),
    }


def _prepare_input(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df_in = pd.DataFrame(records)

    # If user sent absolutely nothing useful:
    overlap = set(df_in.columns) & set(FRIENDLY_FEATURES + [ID_COL])
    if len(overlap) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No valid input fields provided. Expected some of: {FRIENDLY_FEATURES} (+ optional {ID_COL})",
        )

    # Ensure all friendly columns exist (so Streamlit can send partial records)
    for c in FRIENDLY_FEATURES:
        if c not in df_in.columns:
            df_in[c] = np.nan

    # appointment_id is optional, but helpful for UI tables
    if ID_COL not in df_in.columns:
        df_in[ID_COL] = None

    # Coerce numeric friendly
    for c in NUMERIC_FRIENDLY:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce").fillna(0)

    # Clean categorical friendly
    for c in CATEGORICAL_FRIENDLY:
        df_in[c] = (
            df_in[c]
            .astype("string")
            .fillna("Unknown")
            .replace("nan", "Unknown")
            .replace("", "Unknown")
        )

    # Build the model frame exactly as the pipeline expects:
    # IMPORTANT: only MODEL_FEATURES go into the model
    for c in MODEL_FEATURES:
        if c not in df_in.columns:
            df_in[c] = np.nan

    df_model = df_in[MODEL_FEATURES].replace({pd.NA: np.nan})

    return df_in, df_model


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df_raw, df_model = _prepare_input(req.records)

        proba = model.predict_proba(df_model)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)

        # Return appointment_id back so Streamlit can display high-risk rows
        return {
            "threshold": THRESHOLD,
            "predictions": [
                {
                    "appointment_id": None if pd.isna(aid) else str(aid),
                    "no_show_probability": float(p),
                    "predicted_label": int(y),
                }
                for aid, p, y in zip(df_raw[ID_COL].tolist(), proba, pred)
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(e).__name__}: {str(e)}",
        )
    

@app.get("/version")
def version():
    import sys, sklearn, pandas, numpy, joblib
    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    }

    