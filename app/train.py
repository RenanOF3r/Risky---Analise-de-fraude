import argparse
import json
from pathlib import Path

try:
    from app.pipeline import train_and_save
except ModuleNotFoundError:
    # Allows running as `python app/train.py`
    from pipeline import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection model on PaySim dataset.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/paysim.csv",
        help="Path to the PaySim CSV file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="models",
        help="Directory to store model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = train_and_save(args.csv_path, artifacts_dir=args.artifacts_dir)
    metrics_path = Path(artifacts.metrics_path)
    with metrics_path.open() as f:
        metrics = json.load(f)

    print(f"Saved model to {artifacts.model_path}")
    print(f"Saved preprocessor to {artifacts.preprocessor_path}")
    print(f"Evaluation metrics (test set):")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
