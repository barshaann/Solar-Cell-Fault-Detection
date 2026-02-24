from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from solar_fault.config import load_config
from solar_fault.data import make_split
from solar_fault.evaluate import evaluate_binary_classifier, save_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved solar fault model without retraining.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved .keras model. Defaults to <model_dir>/<model_name> from config.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for test metrics JSON (default: <model_dir>/test_metrics.json).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = Path(args.model) if args.model else (cfg.model_dir / cfg.model_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at '{model_path}'. Train once with scripts/train.py, then re-run this script."
        )

    split = make_split(cfg.data)
    model = tf.keras.models.load_model(model_path)

    y_prob = model.predict(split.x_val, verbose=0).ravel()
    metrics = evaluate_binary_classifier(split.y_val.astype(int), y_prob)

    out_path = Path(args.out) if args.out else (cfg.model_dir / "test_metrics.json")
    save_metrics(metrics, out_path)
    print(f"Loaded model: {model_path}")
    print(f"Saved metrics: {out_path}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['average_precision']:.4f}")


if __name__ == "__main__":
    main()
