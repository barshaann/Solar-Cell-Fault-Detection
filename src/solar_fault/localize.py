from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf



def gradcam_heatmap(model: tf.keras.Model, image_batch: np.ndarray) -> np.ndarray:
    if image_batch.ndim != 4 or image_batch.shape[0] != 1:
        raise ValueError("image_batch must have shape (1, H, W, C).")

    last_conv = None
    for layer in reversed(model.layers):
        if len(getattr(layer, "output_shape", ())) == 4:
            last_conv = layer
            break

    if last_conv is None:
        raise RuntimeError("No convolution layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def defect_boxes(heatmap: np.ndarray, threshold: float = 0.45):
    mask = np.uint8((heatmap > threshold) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask



def draw_boxes(image: np.ndarray, contours):
    canvas = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 50, 50), 2)
    return canvas
