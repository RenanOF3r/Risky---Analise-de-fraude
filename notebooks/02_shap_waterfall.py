import argparse
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from app.explain import save_waterfall_plot
from app.pipeline import add_velocity_features_offline, engineer_features, filter_relevant_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera SHAP waterfall para um caso de fraude.")
    parser.add_argument("--csv-path", type=str, default="data/paysim.csv")
    parser.add_argument("--model-path", type=str, default="models/model.joblib")
    parser.add_argument("--out-path", type=str, default="reports/shap_waterfall.png")
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    return parser.parse_args()


def find_first_fraud(csv_path: str, max_rows: int) -> pd.DataFrame:
    usecols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "isFraud",
    ]
    chunksize = 200_000
    read_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        read_rows += len(chunk)
        df = engineer_features(chunk)
        df = filter_relevant_transactions(df)
        frauds = df[df["isFraud"] == 1]
        if not frauds.empty:
            return frauds.head(1)
        if read_rows >= max_rows:
            break
    raise RuntimeError("Não encontrei uma linha fraudulenta dentro do limite de leitura.")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    model = joblib.load(args.model_path)

    fraud_row = find_first_fraud(args.csv_path, max_rows=args.max_rows)
    fraud_row = add_velocity_features_offline(fraud_row)
    X = fraud_row.drop(columns=["isFraud"])

    save_waterfall_plot(model, X, str(out_path), max_display=12)
    print(f"SHAP waterfall salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
