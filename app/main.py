import os
import threading
from collections import deque
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import engineer_features
from app.explain import explain_row_with_shap


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


@lru_cache()
def get_threshold() -> float:
    metrics_path = os.environ.get("RISKY_METRICS_PATH", "models/metrics.json")
    if os.path.exists(metrics_path):
        try:
            import json

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return float(metrics.get("best_threshold", 0.5))
        except Exception:
            return 0.5
    return float(os.environ.get("RISKY_THRESHOLD", 0.5))


class VelocityStore:
    """
    In-memory velocity feature store for demo purposes.

    Keeps a deque of recent `step` values per entity, and computes count in the last 1 hour window
    (current step and previous step, since PaySim `step` is hour-based).
    """

    def __init__(self, max_events_per_entity: int = 5000):
        self._lock = threading.Lock()
        self._orig: dict[str, deque[int]] = {}
        self._dest: dict[str, deque[int]] = {}
        self._max_events_per_entity = max_events_per_entity

    @staticmethod
    def _prune(events: deque[int], current_step: int) -> None:
        min_step = current_step - 1
        while events and events[0] < min_step:
            events.popleft()

    def update_and_count(self, step: int, name_orig: str, name_dest: str) -> tuple[int, int]:
        with self._lock:
            orig_events = self._orig.setdefault(name_orig, deque())
            dest_events = self._dest.setdefault(name_dest, deque())

            self._prune(orig_events, step)
            self._prune(dest_events, step)

            # Count prior events in the last 1h window (exclude current event).
            orig_count = len(orig_events)
            dest_count = len(dest_events)

            orig_events.append(step)
            dest_events.append(step)

            if len(orig_events) > self._max_events_per_entity:
                orig_events.popleft()
            if len(dest_events) > self._max_events_per_entity:
                dest_events.popleft()

            return orig_count, dest_count


velocity_store = VelocityStore()

def build_features(transaction: Transaction) -> pd.DataFrame:
    df = pd.DataFrame([transaction.model_dump()])
    orig_count, dest_count = velocity_store.update_and_count(
        step=transaction.step, name_orig=transaction.nameOrig, name_dest=transaction.nameDest
    )
    df["tx_count_orig_last_1h"] = orig_count
    df["tx_count_dest_last_1h"] = dest_count
    return engineer_features(df)


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

    if transaction.type not in {"TRANSFER", "CASH_OUT"}:
        raise HTTPException(
            status_code=400,
            detail="Transaction type not supported for this model (expected TRANSFER or CASH_OUT).",
        )

    df_features = build_features(transaction)

    try:
        fraud_proba = model.predict_proba(df_features)[:, 1][0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    threshold = get_threshold()
    is_fraud = bool(fraud_proba >= threshold)
    reason_code = derive_reason_code(df_features.iloc[0])

    return {
        "fraud_probability": float(fraud_proba),
        "is_fraud": is_fraud,
        "reason_code": reason_code,
        "threshold": float(threshold),
    }


@app.post("/explain")
def explain(transaction: Transaction, top_k: int = 12):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if transaction.type not in {"TRANSFER", "CASH_OUT"}:
        raise HTTPException(
            status_code=400,
            detail="Transaction type not supported for this model (expected TRANSFER or CASH_OUT).",
        )

    df_features = build_features(transaction)
    try:
        explanation = explain_row_with_shap(model, df_features, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    return {
        "base_value": float(explanation.base_value),
        "top_features": explanation.top_features,
    }
