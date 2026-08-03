import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr

from src.dataset import get_valid_transforms
from src.model import build_model
from src.gradcam import PyTorchGradCAM, overlay_cam_on_image

# Cache loaded models in memory for fast inference
LOADED_MODELS = {}

def get_model(backbone_name):
    backbone_name = backbone_name.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if backbone_name not in LOADED_MODELS:
        model = build_model(model_name=backbone_name, num_classes=2, pretrained=True).to(device)
        weights_path = f"src/best_{backbone_name}.pth"
        if not os.path.exists(weights_path):
            weights_path = "src/best_weights.pth"

        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device))
                print(f"Loaded weights for {backbone_name} from {weights_path}")
            except Exception as e:
                print(f"Warning loading weights for {backbone_name}: {e}")
        else:
            print(f"Using default pretrained weights for {backbone_name}")

        model.eval()
        LOADED_MODELS[backbone_name] = model

    return LOADED_MODELS[backbone_name], device

def analyze_xray(input_img, backbone_name, alpha_transparency):
    if input_img is None:
        return "<div style='color: red;'>Please upload or select a Chest X-Ray image.</div>", None, None

    # Convert PIL Image to RGB numpy array
    if isinstance(input_img, Image.Image):
        rgb_np = np.array(input_img.convert("RGB"))
    else:
        rgb_np = input_img.copy()

    # Preprocess image for model
    transform = get_valid_transforms(img_size=224)
    augmented = transform(image=rgb_np)
    img_tensor = augmented['image'].unsqueeze(0)

    # Get model & device
    model, device = get_model(backbone_name)
    img_tensor = img_tensor.to(device)

    # Generate Grad-CAM activation map
    gradcam = PyTorchGradCAM(model)
    cam_mask, pred_class_idx, confidence = gradcam.generate_cam(img_tensor)

    classes = ["NORMAL", "PNEUMONIA"]
    predicted_class = classes[pred_class_idx]
    normal_prob = 1.0 - confidence if pred_class_idx == 1 else confidence
    pneumonia_prob = confidence if pred_class_idx == 1 else 1.0 - confidence

    # Create resized RGB image for overlay matching Grad-CAM resolution
    resized_rgb = cv2.resize(rgb_np, (224, 224)) / 255.0
    overlay_img, raw_heatmap = overlay_cam_on_image(resized_rgb, cam_mask, alpha=alpha_transparency)

    # Build Markdown Diagnostic Report
    status_color = "#e74c3c" if predicted_class == "PNEUMONIA" else "#2ecc71"
    status_icon = "🚨" if predicted_class == "PNEUMONIA" else "✅"

    assessment_text = (
        "**Clinical Attention Alert**: High-intensity activation detected in focal lung fields, indicating opacity consistent with acute infiltrates or consolidation."
        if predicted_class == "PNEUMONIA" else
        "**Clinical Assessment**: Clear bilateral lung fields with no focal consolidation or acute pulmonary opacity detected."
    )

    report_html = f"""
    <div style='background-color: #1e1e2e; padding: 20px; border-radius: 12px; border: 2px solid {status_color}; font-family: sans-serif; color: #ffffff;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <h2 style='margin: 0; color: {status_color};'>{status_icon} Diagnosis: {predicted_class}</h2>
            <span style='background: {status_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px;'>
                Confidence: {confidence*100:.1f}%
            </span>
        </div>
        <hr style='border: 0.5px solid #333; margin: 15px 0;'/>
        <div style='margin-bottom: 12px;'>
            <strong>Probability Breakdown:</strong>
            <div style='margin-top: 6px; background: #2b2b3d; border-radius: 8px; padding: 8px;'>
                <div style='margin-bottom: 4px;'>Pneumonia: <strong>{pneumonia_prob*100:.1f}%</strong></div>
                <div style='height: 8px; background: #444; border-radius: 4px;'>
                    <div style='height: 100%; width: {pneumonia_prob*100}%; background: #e74c3c; border-radius: 4px;'></div>
                </div>
                <div style='margin-top: 8px; margin-bottom: 4px;'>Normal: <strong>{normal_prob*100:.1f}%</strong></div>
                <div style='height: 8px; background: #444; border-radius: 4px;'>
                    <div style='height: 100%; width: {normal_prob*100}%; background: #2ecc71; border-radius: 4px;'></div>
                </div>
            </div>
        </div>
        <p style='font-size: 13px; line-height: 1.5; color: #dddddd;'>{assessment_text}</p>
        <div style='font-size: 11px; color: #888888; font-style: italic;'>Model Backbone: {backbone_name.upper()} | Grad-CAM Resolution: 224x224</div>
    </div>
    """

    return report_html, overlay_img, raw_heatmap

# Collect Sample Images from dataset for quick click-and-test UI
def load_sample_images():
    samples = []
    normal_dir = "data/test/NORMAL"
    pneumonia_dir = "data/test/PNEUMONIA"

    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir)[:2]:
            if f.endswith(".jpeg"):
                samples.append([os.path.join(normal_dir, f), "convnext", 0.5])
    if os.path.exists(pneumonia_dir):
        for f in os.listdir(pneumonia_dir)[:2]:
            if f.endswith(".jpeg"):
                samples.append([os.path.join(pneumonia_dir, f), "convnext", 0.5])
    return samples

# Build Gradio UI
with gr.Blocks(title="Chest X-Ray Pneumonia Detection AI") as demo:
    gr.Markdown(
        """
        # 🫁 Modern Chest X-Ray Pneumonia Detection & Grad-CAM Visualizer
        ### Deep Learning Transfer Learning (`ConvNeXt` / `ViT`) with Gradient-weighted Class Activation Mapping
        Upload a Pediatric/Adult Chest X-Ray image to generate real-time diagnostic predictions and visual heatmaps highlighting lung opacity regions.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Chest X-Ray Image", sources=["upload", "clipboard"])
            backbone_selector = gr.Dropdown(
                choices=["convnext", "vit", "resnet", "baseline"],
                value="convnext",
                label="Select Vision Architecture Backbone",
                info="Choose modern ConvNeXt, Vision Transformer (ViT), ResNet-50, or Baseline CNN"
            )
            alpha_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                label="Grad-CAM Heatmap Transparency",
                info="Adjust overlay blending transparency"
            )
            analyze_btn = gr.Button("🔍 Run Diagnostic Inference & Grad-CAM", variant="primary", size="lg")

        with gr.Column(scale=1):
            diagnostic_output = gr.HTML(label="Diagnostic Assessment & Confidence")
            with gr.Tabs():
                with gr.TabItem("Grad-CAM Overlay"):
                    cam_overlay = gr.Image(label="Lung Opacity Attention Map (Grad-CAM)")
                with gr.TabItem("Raw Heatmap"):
                    raw_heatmap_out = gr.Image(label="Raw Class Activation Heatmap")

    sample_list = load_sample_images()
    if sample_list:
        gr.Examples(
            examples=sample_list,
            inputs=[input_image, backbone_selector, alpha_slider],
            outputs=[diagnostic_output, cam_overlay, raw_heatmap_out],
            fn=analyze_xray,
            cache_examples=False,
            label="Try Sample Chest X-Rays"
        )

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[input_image, backbone_selector, alpha_slider],
        outputs=[diagnostic_output, cam_overlay, raw_heatmap_out]
    )

    gr.Markdown(
        """
        ---
        **Disclaimer**: *This AI diagnostic tool is designed for research, portfolio demonstration, and decision-support exploration only. All predictions should be verified by certified radiologists.*
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
