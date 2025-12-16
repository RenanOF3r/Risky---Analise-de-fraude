import os
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import engineer_features


class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Hour step since start of simulation")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., gt=0, description="Transaction amount")
    nameOrig: str
    oldbalanceOrg: float = Field(..., description="Balance before the transaction for origin account")
    newbalanceOrig: float = Field(..., description="Balance after the transaction for origin account")
    nameDest: str
    oldbalanceDest: float = Field(..., description="Balance before the transaction for destination account")
    newbalanceDest: float = Field(..., description="Balance after the transaction for destination account")
    isFlaggedFraud: int | None = Field(
        0, description="Flag set by legacy rules, optional", ge=0, le=1
    )


app = FastAPI(title="Risky - Fraud Detection API", version="0.1.0")


@lru_cache()
def get_model():
    model_path = os.environ.get("RISKY_MODEL_PATH", "models/model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train a model with `python -m app.train` first."
        )
    return joblib.load(model_path)


def derive_reason_code(row: pd.Series) -> str:
    reasons = []
    if row["amount"] > 100000:
        reasons.append("HIGH_AMOUNT")
    if row.get("balance_error_orig", 0) > 1e3:
        reasons.append("ORIG_BAL_MISMATCH")
    if row.get("balance_error_dest", 0) < -1e3:
        reasons.append("DEST_BAL_MISMATCH")
    if row["type"] not in {"TRANSFER", "CASH_OUT"}:
        reasons.append("LOW_RISK_TYPE")
    return reasons[0] if reasons else "MODEL_SCORE"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    df = pd.DataFrame([transaction.model_dump()])
    df_features = engineer_features(df)

    if df_features.empty:
        raise HTTPException(
            status_code=400,
            detail="Transaction type not supported for this model (expected TRANSFER or CASH_OUT).",
        )

    try:
        fraud_proba = model.predict_proba(df_features)[:, 1][0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    is_fraud = bool(fraud_proba >= 0.5)
    reason_code = derive_reason_code(df_features.iloc[0])

    return {
        "fraud_probability": float(fraud_proba),
        "is_fraud": is_fraud,
        "reason_code": reason_code,
    }
