from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

from .config import TrainConfig



def build_model(image_size: int, train_cfg: TrainConfig) -> Model:
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(train_cfg.dropout)(x)
    x = layers.Dense(train_cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(train_cfg.dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model



def fine_tune(model: Model, unfreeze_last_n_layers: int = 40, learning_rate: float = 1e-5) -> None:
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Backbone model not found for fine-tuning.")

    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_last_n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
