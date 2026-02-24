from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

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



def _paths_to_arrays(train_paths: list[str], val_paths: list[str], df: pd.DataFrame, image_size: int) -> DatasetSplit:
    label_map = {row.full_path: int(row.label) for row in df.itertuples(index=False)}

    x_train = np.stack([load_image(p, image_size) for p in train_paths], axis=0)
    x_val = np.stack([load_image(p, image_size) for p in val_paths], axis=0)
    y_train = np.array([label_map[p] for p in train_paths], dtype=np.float32)
    y_val = np.array([label_map[p] for p in val_paths], dtype=np.float32)

    return DatasetSplit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)



def make_split(cfg: DataConfig) -> DatasetSplit:
    df = load_dataframe(cfg)
    paths = df["full_path"].values
    labels = df["label"].values

    x_train_path, x_val_path, _, _ = train_test_split(
        paths,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=labels,
    )

    return _paths_to_arrays(x_train_path.tolist(), x_val_path.tolist(), df, cfg.image_size)



def save_split_manifest(path: str | Path, train_paths: list[str], val_paths: list[str]) -> None:
    payload = {"train_paths": train_paths, "val_paths": val_paths}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))



def build_split_from_manifest(cfg: DataConfig, manifest_path: str | Path) -> DatasetSplit:
    manifest = json.loads(Path(manifest_path).read_text())
    train_paths = manifest["train_paths"]
    val_paths = manifest["val_paths"]

    df = load_dataframe(cfg)
    return _paths_to_arrays(train_paths, val_paths, df, cfg.image_size)



def split_paths(cfg: DataConfig) -> tuple[list[str], list[str]]:
    df = load_dataframe(cfg)
    paths = df["full_path"].values
    labels = df["label"].values
    train_paths, val_paths, _, _ = train_test_split(
        paths,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=labels,
    )
    return train_paths.tolist(), val_paths.tolist()



def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.astype(int))
    return {int(c): float(w) for c, w in zip(classes, weights)}
