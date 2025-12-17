from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _try_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def _dense(matrix_like):
    if hasattr(matrix_like, "toarray"):
        return matrix_like.toarray()
    return matrix_like


def _get_feature_names(preprocessor) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    return []


@dataclass
class ShapExplanation:
    base_value: float
    top_features: list[dict[str, Any]]


def explain_row_with_shap(
    model_pipeline,
    df_features: pd.DataFrame,
    top_k: int = 12,
) -> ShapExplanation:
    """
    Compute SHAP contributions for a single row.

    Requires `shap` to be installed (it is installed in Docker image with Python < 3.13).
    """
    shap = _try_import_shap()
    if shap is None:
        raise RuntimeError(
            "SHAP is not available in this environment. "
            "On Windows + Python 3.13, install build tools or run via Docker (Python 3.11)."
        )

    if len(df_features) != 1:
        raise ValueError("df_features must contain exactly 1 row")

    preprocessor = model_pipeline.named_steps["preprocess"]
    model = model_pipeline.named_steps["model"]

    X_trans = preprocessor.transform(df_features)
    X_dense = _dense(X_trans)
    feature_names = _get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(model)

    try:
        explanation = explainer(X_dense)
        shap_values = explanation.values[0]
        base_value = explanation.base_values[0]
        data_row = explanation.data[0]
    except Exception:
        shap_values = explainer.shap_values(X_dense)[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.array(base_value).ravel()[0])
        data_row = np.array(X_dense)[0]

    shap_values = np.array(shap_values).ravel()
    if feature_names and len(feature_names) == len(shap_values):
        names = feature_names
    else:
        names = [f"f{i}" for i in range(len(shap_values))]

    abs_order = np.argsort(np.abs(shap_values))[::-1][:top_k]
    top = []
    for idx in abs_order:
        top.append(
            {
                "feature": names[int(idx)],
                "value": float(np.array(data_row).ravel()[int(idx)]),
                "shap_value": float(shap_values[int(idx)]),
            }
        )

    return ShapExplanation(base_value=float(base_value), top_features=top)


def save_waterfall_plot(
    model_pipeline,
    df_features: pd.DataFrame,
    output_path: str,
    max_display: int = 12,
) -> None:
    shap = _try_import_shap()
    if shap is None:
        raise RuntimeError(
            "SHAP is not available in this environment. "
            "On Windows + Python 3.13, run this via Docker (Python 3.11)."
        )

    import matplotlib.pyplot as plt

    preprocessor = model_pipeline.named_steps["preprocess"]
    model = model_pipeline.named_steps["model"]

    X_dense = _dense(preprocessor.transform(df_features))
    feature_names = _get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_dense)

    if feature_names:
        explanation.feature_names = feature_names

    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

