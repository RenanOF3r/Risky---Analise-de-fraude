import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from app.pipeline import engineer_features, filter_relevant_transactions, optimize_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA orientada para o dataset PaySim.")
    parser.add_argument("--csv-path", type=str, default="data/paysim.csv")
    parser.add_argument("--out-dir", type=str, default="reports/eda")
    parser.add_argument("--sample-rows", type=int, default=250_000)
    return parser.parse_args()


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "isFraud",
        "isFlaggedFraud",
    ]
    dtypes = {
        "step": "int32",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }

    df_raw = pd.read_csv(args.csv_path, usecols=usecols, dtype=dtypes, nrows=args.sample_rows)
    mem_before = memory_mb(df_raw)
    df = optimize_types(df_raw.copy())
    mem_after = memory_mb(df)

    fraud_rate = float(df["isFraud"].mean())
    null_accuracy = 1.0 - fraud_rate

    print("=== Resumo (amostra) ===")
    print(f"Linhas: {len(df):,}")
    print(f"Memória (antes): {mem_before:.1f} MB | (depois): {mem_after:.1f} MB")
    print(f"Fraude (%): {fraud_rate*100:.4f}%")
    print(f"Null accuracy (sempre legítimo): {null_accuracy*100:.4f}%")

    sns.set_theme(style="whitegrid")

    # 1) Desbalanceamento
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="isFraud", data=df)
    ax.set_title("Desbalanceamento: fraudes vs legítimas (amostra)")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("count")
    savefig(out_dir / "01_class_imbalance.png")

    # 2) Fraude por tipo
    fraud_by_type = (
        df.groupby("type")["isFraud"].mean().sort_values(ascending=False).reset_index(name="fraud_rate")
    )
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=fraud_by_type, x="type", y="fraud_rate")
    ax.set_title("Taxa de fraude por tipo de transação (amostra)")
    ax.set_xlabel("type")
    ax.set_ylabel("fraud_rate")
    savefig(out_dir / "02_fraud_rate_by_type.png")

    # 3) Features de domínio (hora do dia e erro de saldo)
    df_feat = engineer_features(df)
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df_feat.groupby("hour_of_day")["isFraud"].mean().reset_index(),
        x="hour_of_day",
        y="isFraud",
    )
    plt.title("Taxa de fraude por hora do dia (hour_of_day)")
    plt.xlabel("hour_of_day")
    plt.ylabel("fraud_rate")
    savefig(out_dir / "03_fraud_rate_by_hour.png")

    # 4) Montante: distribuição (log) por classe para tipos de risco
    df_risky = filter_relevant_transactions(df_feat)
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=df_risky,
        x="amount",
        hue="isFraud",
        bins=60,
        log_scale=(True, False),
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Distribuição do amount (log-x) em TRANSFER/CASH_OUT")
    plt.xlabel("amount (log)")
    plt.ylabel("density")
    savefig(out_dir / "04_amount_distribution_risky_types.png")

    # 5) Erro de saldo: comparação (boxplot) para fraudes vs legítimas
    sample_bal = df_risky.sample(min(len(df_risky), 20_000), random_state=42)
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=sample_bal, x="isFraud", y="balance_error_orig")
    plt.title("balance_error_orig por classe (amostra reduzida)")
    plt.xlabel("isFraud")
    plt.ylabel("balance_error_orig")
    savefig(out_dir / "05_balance_error_orig_boxplot.png")

    print(f"Gráficos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
