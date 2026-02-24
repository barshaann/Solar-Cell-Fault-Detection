from __future__ import annotations

import argparse

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

from solar_fault.localize import gradcam_heatmap, defect_boxes, draw_boxes



def parse_args():
    p = argparse.ArgumentParser(description="Launch fault detection app")
    p.add_argument("--model", type=str, default="artifacts/solar_fault_detector.keras")
    return p.parse_args()



def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    img = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)



def build_interface(model: tf.keras.Model):
    def analyze_image(image: np.ndarray):
        if image is None:
            return None, None, None, "No image provided."

        resized = cv2.resize(image, (224, 224))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        prob = float(model.predict(x, verbose=0)[0][0])
        cls = "FAULT" if prob >= 0.5 else "NO FAULT"

        heat = gradcam_heatmap(model, x)
        heat = cv2.resize(heat, (224, 224))
        heat_rgb = heatmap_to_rgb(heat)
        overlay = cv2.addWeighted(resized, 0.6, heat_rgb, 0.4, 0)
        contours, _ = defect_boxes(heat)
        boxed = draw_boxes(resized, contours)

        return heat_rgb, overlay, boxed, f"{cls} ({prob:.3f})"

    return gr.Interface(
        fn=analyze_image,
        inputs=gr.Image(type="numpy", label="Upload EL image"),
        outputs=[
            gr.Image(type="numpy", label="Grad-CAM heatmap"),
            gr.Image(type="numpy", label="Overlay"),
            gr.Image(type="numpy", label="Defect boxes"),
            gr.Textbox(label="Prediction"),
        ],
        title="Solar Cell Fault Detection & Localization",
    )



def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    demo = build_interface(model)
    demo.launch()


if __name__ == "__main__":
    main()
