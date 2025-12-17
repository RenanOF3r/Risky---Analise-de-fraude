import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
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


def add_velocity_features_offline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate "velocity" features for offline training:
    number of transactions for the same account in the current hour + previous hour.

    Note: PaySim uses `step` as hours. This keeps the implementation efficient for large CSVs.
    """
    df = df.copy()

    def compute_last_1h_count(entity_col: str) -> pd.Series:
        step_counts = (
            df.groupby([entity_col, "step"], sort=False)
            .size()
            .reset_index(name="step_count")
            .sort_values([entity_col, "step"])
        )
        step_counts["prev_step"] = step_counts.groupby(entity_col)["step"].shift(1)
        step_counts["prev_count"] = step_counts.groupby(entity_col)["step_count"].shift(1).fillna(0)
        step_counts["prev_count"] = np.where(
            (step_counts["step"] - step_counts["prev_step"]) == 1, step_counts["prev_count"], 0
        )
        step_counts["last_1h_count"] = step_counts["step_count"] + step_counts["prev_count"]
        merged = df[[entity_col, "step"]].merge(
            step_counts[[entity_col, "step", "last_1h_count"]],
            on=[entity_col, "step"],
            how="left",
        )
        # Exclude the current transaction itself (approximation per step bucket).
        return np.maximum(merged["last_1h_count"].fillna(0).astype("int32") - 1, 0).astype("int32")

    df["tx_count_orig_last_1h"] = compute_last_1h_count("nameOrig")
    df["tx_count_dest_last_1h"] = compute_last_1h_count("nameDest")
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
        "tx_count_orig_last_1h",
        "tx_count_dest_last_1h",
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
) -> Tuple[ImbPipeline, dict, np.ndarray, np.ndarray]:
    df = optimize_types(df)
    df = filter_relevant_transactions(df)
    df = engineer_features(df)
    df = add_velocity_features_offline(df)

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

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    if len(thresholds) > 0:
        best_idx = int(np.nanargmax(f1_scores[:-1]))
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    tuning_thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    tuning_rows = []
    for t in tuning_thresholds:
        pred = (y_proba >= t).astype(int)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        tuning_rows.append(
            {
                "threshold": float(t),
                "precision": float(report["1"]["precision"]),
                "recall": float(report["1"]["recall"]),
                "f1": float(report["1"]["f1-score"]),
            }
        )

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "auprc": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_threshold": float(best_threshold),
        "threshold_tuning": tuning_rows,
        "scale_pos_weight": scale_pos_weight,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "test_size": test_size,
    }

    return clf, metrics, np.array(y_test), np.array(y_proba)


def save_evaluation_reports(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: str = "reports",
    threshold: float | None = None,
) -> None:
    import csv
    import os

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    if threshold is None:
        threshold = 0.5

    # Confusion matrix plot
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=160)
    plt.close(fig)

    # Threshold tuning table (CSV)
    tuning_thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for t in tuning_thresholds:
        pred = (y_proba >= t).astype(int)
        report = classification_report(y_true, pred, output_dict=True, zero_division=0)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(report["1"]["precision"]),
                "recall": float(report["1"]["recall"]),
                "f1": float(report["1"]["f1-score"]),
            }
        )

    csv_path = os.path.join(out_dir, "threshold_tuning.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


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


def train_and_save(csv_path: str, artifacts_dir: str = "models", reports_dir: str = "reports") -> ModelArtifacts:
    df = load_dataset(csv_path)
    model, metrics, y_true, y_proba = train_model(df)
    artifacts = persist_artifacts(model, metrics, artifacts_dir=artifacts_dir)
    save_evaluation_reports(
        y_true=y_true,
        y_proba=y_proba,
        out_dir=reports_dir,
        threshold=float(metrics.get("best_threshold", 0.5)),
    )
    return artifacts
