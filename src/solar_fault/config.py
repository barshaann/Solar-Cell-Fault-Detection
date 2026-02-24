from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Literal


UncertainPolicy = Literal["drop", "fault", "non_fault"]


@dataclass
class DataConfig:
    dataset_root: Path = Path("dataset")
    labels_file: str = "labels.csv"
    image_dir: str = "images"
    image_size: int = 224
    split_seed: int = 42
    test_size: float = 0.2
    uncertain_policy: UncertainPolicy = "drop"
    fault_threshold: float = 0.66
    non_fault_threshold: float = 0.33


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    dropout: float = 0.3
    dense_units: int = 256
    patience: int = 8
    mixed_precision: bool = False


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model_dir: Path = Path("artifacts")
    model_name: str = "solar_fault_detector.keras"



def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    payload = json.loads(Path(path).read_text())
    data = DataConfig(**payload.get("data", {}))
    train = TrainConfig(**payload.get("train", {}))
    config = AppConfig(
        data=data,
        train=train,
        model_dir=Path(payload.get("model_dir", "artifacts")),
        model_name=payload.get("model_name", "solar_fault_detector.keras"),
    )
    return config
