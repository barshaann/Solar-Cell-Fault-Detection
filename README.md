# Solar Cell Fault Detection & Localization

This repository now includes a production-style training and inference pipeline for binary fault detection on EL solar-cell imagery, with explainability and localization support.

## What is included

- Config-driven training (`config.example.json`).
- Deterministic data split and clean label-policy handling.
- Class-weighted training to handle imbalance.
- Transfer learning with EfficientNetV2 and optional fine-tuning.
- Validation metrics export (`artifacts/metrics.json`).
- Gradio app for prediction + Grad-CAM + defect box visualization.

## Project structure

- `src/solar_fault/config.py` – schema and config loading.
- `src/solar_fault/data.py` – CSV parsing, label mapping, image loading, split.
- `src/solar_fault/model.py` – model creation and fine-tuning utility.
- `src/solar_fault/evaluate.py` – evaluation metrics and persistence.
- `src/solar_fault/localize.py` – Grad-CAM and contour-based localization.
- `scripts/train.py` – training entrypoint.
- `scripts/app.py` – demo app entrypoint.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Train

```bash
python scripts/train.py --config config.example.json --fine-tune
```

## Run app

```bash
python scripts/app.py --model artifacts/solar_fault_detector.keras
```

## Notes for higher accuracy

1. Keep `uncertain_policy="drop"` for clean labels.
2. Increase epochs and unfreeze more backbone layers only after baseline stabilizes.
3. Track AUC + PR-AUC on a locked validation set.
4. Add hard-negative mining and TTA for inference if latency budget allows.
