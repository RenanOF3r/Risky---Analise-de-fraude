import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass
class ModelArtifacts:
    preprocessor_path: str
    model_path: str
    metrics_path: str


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def filter_relevant_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only transaction types that historically carry fraud risk."""
    risky_types = {"TRANSFER", "CASH_OUT"}
    return df[df["type"].isin(risky_types)].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features for fraud detection."""
    df = df.copy()
    df["hour_of_day"] = df["step"] % 24
    df["balance_error_orig"] = (
        df["oldbalanceOrg"].fillna(0) + df["amount"].fillna(0) - df["newbalanceOrig"].fillna(0)
    )
    df["balance_error_dest"] = (
        df["oldbalanceDest"].fillna(0) + df["amount"].fillna(0) - df["newbalanceDest"].fillna(0)
    )
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").fillna(False)
    df["orig_is_merchant"] = df["nameOrig"].str.startswith("M").fillna(False)
    return df


def build_preprocessor(feature_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_features = ["type", "dest_is_merchant", "orig_is_merchant"]
    numeric_features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "hour_of_day",
        "balance_error_orig",
        "balance_error_dest",
    ]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor, categorical_features, numeric_features


def train_model(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ImbPipeline, dict]:
    df = optimize_types(df)
    df = filter_relevant_transactions(df)
    df = engineer_features(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor, categorical_features, numeric_features = build_preprocessor(X)

    train_df, test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    positive = y_train.sum()
    negative = len(y_train) - positive
    scale_pos_weight = max(1.0, negative / max(positive, 1))

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=random_state,
    )

    clf = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(train_df, y_train)

    y_proba = clf.predict_proba(test_df)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "auprc": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "scale_pos_weight": scale_pos_weight,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "test_size": test_size,
    }

    return clf, metrics


def persist_artifacts(model: ImbPipeline, metrics: dict, artifacts_dir: str = "models") -> ModelArtifacts:
    os.makedirs(artifacts_dir, exist_ok=True)
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
    model_path = os.path.join(artifacts_dir, "model.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    # Persist both full pipeline and the fitted preprocessor separately for clarity
    from joblib import dump

    dump(model.named_steps["preprocess"], preprocessor_path)
    dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return ModelArtifacts(
        preprocessor_path=preprocessor_path, model_path=model_path, metrics_path=metrics_path
    )


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    expected_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    return df


def train_and_save(csv_path: str, artifacts_dir: str = "models") -> ModelArtifacts:
    df = load_dataset(csv_path)
    model, metrics = train_model(df)
    return persist_artifacts(model, metrics, artifacts_dir=artifacts_dir)
