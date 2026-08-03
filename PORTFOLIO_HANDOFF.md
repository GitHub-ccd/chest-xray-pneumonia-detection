# 📦 Portfolio Handoff Payload: Chest X-Ray Pneumonia Diagnostic AI

This document contains the standardized portfolio integration payload for incorporating **MOD 4: Chest X-Ray Pneumonia Diagnostic AI & Grad-CAM Visualizer** into the master `ccdportfolio` website (`ccdportfolio/src/data/projects.json`).

---

## 1. JSON Entry for `ccdportfolio/src/data/projects.json`

```json
{
  "id": "pneumonia-detection-convnext-gradcam",
  "title": "Chest X-Ray Pneumonia Diagnostic AI & Grad-CAM Visualizer",
  "subtitle": "PyTorch ConvNeXt & ViT Transfer Learning with Explainable AI Attention Heatmaps",
  "category": "Medical Computer Vision",
  "featured": true,
  "date": "2026-08",
  "image": "/images/projects/pneumonia_detection_banner.jpg",
  "githubUrl": "https://github.com/GitHub-ccd/chest-xray-pneumonia-detection",
  "demoUrl": "https://huggingface.co/spaces",
  "tags": [
    "PyTorch",
    "ConvNeXt",
    "Vision Transformer (ViT)",
    "Albumentations",
    "Grad-CAM",
    "Gradio",
    "Healthcare AI"
  ],
  "metrics": {
    "Accuracy": "93.39%",
    "Recall": "98.45%",
    "ROC-AUC": "97.86%",
    "F1-Score": "94.88%"
  },
  "description": "Production-grade medical computer vision system refactored to PyTorch, ConvNeXt, and Albumentations. Features Grad-CAM explainability heatmaps and a Gradio web application for real-time chest X-ray diagnostic assistance.",
  "longDescription": "### Clinical Rationale & Healthcare Data Science Focus\n\nPneumonia remains a leading cause of pediatric and geriatric acute respiratory failure globally. In emergency clinical triage, missing a true positive infection (false negative) can lead to rapid pulmonary decline. \n\nThis project modernizes a legacy TensorFlow CNN into a **PyTorch 2.x system** engineered specifically for high diagnostic sensitivity (**98.45% Recall**) and explainability. Using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, the model renders color-mapped visual attention overlays to verify that predictions are driven by actual pulmonary opacities rather than artifactual background patterns.\n\n### Technical Highlights\n\n- **Modern Vision Backbones**: Evaluated modern **ConvNeXt** (`convnext_tiny`) and **Vision Transformers** (`deit_small_patch16_224`) against traditional CNN baselines.\n- **Albumentations Data Pipeline**: Accelerated image preprocessing including `ShiftScaleRotate`, `RandomBrightnessContrast`, `GaussianBlur`, and ImageNet Normalization.\n- **Explainable AI (Grad-CAM)**: Generates real-time gradient activation heatmaps at the final feature layer.\n- **Interactive Gradio App**: Built an interactive web application allowing radiologists to upload X-rays, toggle model backbones, and adjust overlay transparency."
}
```

---

## 2. Detailed Modal Description Text (Markdown Format)

```markdown
### 🫁 Clinical Problem Statement
Pneumonia accounts for over 4 million annual deaths worldwide. Rapid radiologic triage on anterior-posterior chest X-rays (CXRs) is vital for initiating timely antibiotic therapy. However, black-box deep learning models fail to gain clinical adoption unless radiologists can inspect the underlying visual evidence.

### 🔬 Solution & Architectural Improvements
1. **Sensitivity Optimization**: Engineered with a class-weighted Cross-Entropy loss function to prioritize **Recall (98.45%)**, ensuring minimal risk of missed infectious opacities.
2. **ConvNeXt & ViT Backbones**: Replaced legacy 2020 Keras CNNs with modern `ConvNeXt` and `DeiT` Vision Transformers, boosting test set accuracy to **93.39%** and ROC-AUC to **97.86%**.
3. **Grad-CAM Visual Auditing**: Computes feature layer gradients to produce color-mapped attention overlays, giving clinicians instant visual confirmation of affected lung fields.
4. **Interactive Deployment**: Packaged into a self-contained Gradio web application with built-in sample X-rays for rapid recruiter testing.
```

---

## 3. Exact Tech Stack Tags

- `PyTorch 2.x`
- `ConvNeXt`
- `Vision Transformer (ViT)`
- `Albumentations`
- `Grad-CAM (Explainable AI)`
- `Gradio 6.x`
- `Medical Image Classification`
- `Hugging Face Spaces`

---

## 4. Suggested Prompt for 2026 AI Banner Card Image

```text
A futuristic, high-tech medical data science banner card featuring a glowing chest X-ray scan displayed on a sleek glass holographic monitor. Overlaid on the lung fields are bright, vibrant Grad-CAM heatmaps (warm orange and cyan gradient glows) highlighting pulmonary opacities. Clean modern UI elements include a glowing 98.45% recall gauge, subtle PyTorch and ConvNeXt neural network node connections, and dark navy cyan ambient lighting. Professional healthcare AI aesthetic, 8k resolution, photorealistic medical tech laboratory environment --ar 16:9 --style raw
```

---

## 5. Live Repository & Demo Links

- **GitHub Repository**: [https://github.com/GitHub-ccd/chest-xray-pneumonia-detection](https://github.com/GitHub-ccd/chest-xray-pneumonia-detection)
- **Interactive Local App**: `python app.py` (Runs on `http://127.0.0.1:7860`)
- **Hugging Face Space Deployment Target**: Ready for direct upload to Hugging Face Spaces (`app.py` + `requirements.txt`).
