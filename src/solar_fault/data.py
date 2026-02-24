from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import DataConfig


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray



def _label_from_probability(prob: float, cfg: DataConfig) -> int | None:
    if prob >= cfg.fault_threshold:
        return 1
    if prob <= cfg.non_fault_threshold:
        return 0

    if cfg.uncertain_policy == "drop":
        return None
    if cfg.uncertain_policy == "fault":
        return 1
    return 0



def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    labels_path = Path(cfg.dataset_root) / cfg.labels_file
    df = pd.read_csv(labels_path, sep=" ", header=None, names=["path", "probability", "module_type"])
    df["label"] = df["probability"].map(lambda p: _label_from_probability(float(p), cfg))
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(np.int32)
    df["full_path"] = df["path"].map(lambda p: str((Path(cfg.dataset_root) / p).resolve()))
    return df



def load_image(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr



def make_split(cfg: DataConfig) -> DatasetSplit:
    df = load_dataframe(cfg)
    paths = df["full_path"].values
    labels = df["label"].values

    x_train_path, x_val_path, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=labels,
    )

    x_train = np.stack([load_image(p, cfg.image_size) for p in x_train_path], axis=0)
    x_val = np.stack([load_image(p, cfg.image_size) for p in x_val_path], axis=0)

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train.astype(np.float32),
        x_val=x_val,
        y_val=y_val.astype(np.float32),
    )



def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.astype(int))
    return {int(c): float(w) for c, w in zip(classes, weights)}
