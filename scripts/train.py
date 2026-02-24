from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from solar_fault.config import load_config
from solar_fault.data import make_split, class_weights, save_split_manifest
from solar_fault.model import build_model, fine_tune
from solar_fault.evaluate import evaluate_binary_classifier, save_metrics



def parse_args():
    p = argparse.ArgumentParser(description="Train solar cell fault detector.")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    p.add_argument("--fine-tune", action="store_true", help="Enable fine tuning stage.")
    return p.parse_args()



def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    if cfg.train.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    split = make_split(cfg.data)
    save_split_manifest(cfg.model_dir / "split_manifest.json", split.train_paths, split.val_paths)

    weights = class_weights(split.y_train)
    sample_weights = np.array([weights[int(y)] for y in split.y_train.astype(int)], dtype=np.float32)

    train_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )

    model = build_model(cfg.data.image_size, cfg.train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg.train.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.model_dir / cfg.model_name),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
    ]

    train_ds = tf.data.Dataset.from_tensor_slices((split.x_train, split.y_train, sample_weights))
    train_ds = train_ds.shuffle(len(split.y_train), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(cfg.train.batch_size)
    train_ds = train_ds.map(
        lambda x, y, sw: (train_aug(x, training=True), y, sw),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((split.x_val, split.y_val)).batch(cfg.train.batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    if args.fine_tune:
        fine_tune(model)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(10, cfg.train.epochs // 2),
            callbacks=callbacks,
            verbose=2,
        )

    best = tf.keras.models.load_model(cfg.model_dir / cfg.model_name)
    y_prob = best.predict(split.x_val, verbose=0).ravel()
    metrics = evaluate_binary_classifier(split.y_val.astype(int), y_prob)
    save_metrics(metrics, Path(cfg.model_dir) / "metrics.json")


if __name__ == "__main__":
    main()
