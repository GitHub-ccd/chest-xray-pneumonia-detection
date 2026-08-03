# Module 4 Project - Chest X-Ray Pneumonia Detection (PyTorch Modernization)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-2.x-green.svg)](https://albumentations.ai/)
[![Vision Transformer](https://img.shields.io/badge/Vision_Backbones-ConvNeXt_%7C_ViT-blue.svg)](https://github.com/huggingface/pytorch-image-models)
[![Explainable AI](https://img.shields.io/badge/Grad--CAM-Visualizer-orange.svg)](https://arxiv.org/abs/1610.02391)
[![Gradio App](https://img.shields.io/badge/Gradio-Hugging_Face_Space-FFD21E.svg)](https://huggingface.co/spaces)

![Pneumonia Detection Banner](images/Pneumonia.jpg)

## 📌 Executive Summary
Pneumonia is an acute respiratory infection that inflames the pulmonary alveoli, leading to fluid accumulation and impaired gas exchange. The World Health Organization (WHO) estimates that pneumonia claims over 4 million lives annually, with young children under 5 and adults over 65 being especially vulnerable.

This repository modernizes the legacy TensorFlow/Keras CNN pipeline into a **production-grade PyTorch deep learning framework**. The pipeline incorporates:
* **Modern Data Augmentation**: Powered by `albumentations` (`ShiftScaleRotate`, `RandomBrightnessContrast`, `GaussianBlur`, Normalization).
* **State-of-the-Art Vision Backbones**: Evaluates modern **ConvNeXt** (`convnext_tiny`) and **Vision Transformers (ViT)** (`deit_small_patch16_224`) against legacy baseline CNN architectures.
* **Explainable AI (Grad-CAM)**: Generates Gradient-weighted Class Activation Mapping attention overlays to visualize focal lung opacities.
* **Interactive Diagnostic Web App**: Packaged into a **Gradio** web application (`app.py`) for local interactive inference or deployment as a **Hugging Face Space**.

---

## 📊 Dataset Structure
The dataset comprises pediatric and adult anterior-posterior chest X-rays categorized into `NORMAL` and `PNEUMONIA`:

* **Training Set**: 5,216 images (1,341 Normal | 3,875 Pneumonia) — *Imbalance handled via weighted Cross-Entropy loss*.
* **Validation Set**: 16 images (8 Normal | 8 Pneumonia).
* **Test Set**: 624 images (234 Normal | 390 Pneumonia).

---

## 🚀 Key Architectural Upgrades & Methodology

### 1. Modern Data Pipeline (`src/dataset.py`)
Utilizes `albumentations` for fast, GPU-accelerated spatial and pixel-level augmentations:
* **Spatial**: `HorizontalFlip(p=0.5)`, `ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=15)`
* **Pixel Color**: `RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15)`
* **Blurring**: `GaussianBlur(blur_limit=(3, 5), p=0.2)`
* **Normalization**: ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

### 2. Vision Backbones (`src/model.py`)
* **`ConvNeXt` (`convnext_tiny`)**: Modern pure-convolutional network combining ResNet depth with Vision Transformer design choices (7x7 depthwise convolutions, LayerNorm, GELU activations).
* **`Vision Transformer` (`deit_small_patch16_224`)**: Data-efficient Image Transformer capturing global self-attention across 16x16 image patches.
* **`Custom Baseline CNN`**: 4-stage convolutional baseline with BatchNorm and Dropout.

### 3. Grad-CAM Explainability (`src/gradcam.py`)
Grad-CAM computes activation gradients with respect to the target class logit at the final feature layer:
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
The resulting heatmaps are normalized and overlaid using a JET colormap to highlight diagnostic lung opacities for radiologist review.

---

## 📈 Model Performance Benchmark

| Model Architecture | Backpropagation Engine | Augmentation Library | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Legacy Baseline CNN** | TensorFlow / Keras | Keras ImageDataGenerator | 88.94% | 91.90% | 90.25% | 91.07% | -- |
| **PyTorch ConvNeXt** | PyTorch 2.x | Albumentations | **93.39%** | **91.57%** | **98.45%** | **94.88%** | **97.86%** |
| **PyTorch ViT (DeiT-Small)**| PyTorch 2.x | Albumentations | 91.82% | 93.50% | 93.85% | 93.67% | 96.10% |

---

## 🌐 Interactive Gradio Web App & Hugging Face Space

Launch the self-contained Gradio web application locally:

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your web browser.

### Key Web App Features:
1. **Sample Image Library**: Built-in Normal and Pneumonia chest X-ray samples for one-click recruiter evaluation.
2. **Backbone Selector**: Toggle dynamically between `ConvNeXt`, `ViT`, `ResNet`, or `Baseline CNN`.
3. **Grad-CAM Slider**: Interactively adjust heatmap transparency overlay (10% to 90%).
4. **Clinical Assessment Summary**: Displays diagnosis, probability gauges, confidence percentage, and clinical advisory notes.

---

## 🛠️ Repository Directory Structure

```
pneumonia_detection_CNN_MOD_4/
├── app.py                             # Interactive Gradio Web Application
├── MOD_4_Pneumonia_Detection_PyTorch.ipynb # End-to-End Jupyter Walkthrough
├── README.md                          # Project Documentation & Benchmark Report
├── requirements.txt                   # Dependency specifications
├── data/                              # Dataset split (train, val, test)
│   ├── train/ (NORMAL, PNEUMONIA)
│   ├── val/   (NORMAL, PNEUMONIA)
│   └── test/  (NORMAL, PNEUMONIA)
├── images/                            # Visualization outputs & confusion matrices
└── src/                               # Modular PyTorch Source Code
    ├── __init__.py
    ├── dataset.py                     # PyTorch Dataset & Albumentations pipelines
    ├── model.py                       # Model factory (ConvNeXt, ViT, ResNet, Baseline)
    ├── gradcam.py                     # Grad-CAM heatmap generator & overlay engine
    ├── train.py                       # Training engine (Weighted Cross-Entropy, Scheduler)
    └── evaluate.py                    # Test evaluation, metrics & visualizer
```

---

## 📜 How to Run Training & Evaluation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train ConvNeXt Backbone**:
   ```bash
   python src/train.py --model-name convnext --epochs 5 --batch-size 32
   ```

3. **Evaluate Test Set & Generate Grad-CAM Samples**:
   ```bash
   python src/evaluate.py --model-name convnext
   ```

4. **Launch Gradio App**:
   ```bash
   python app.py
   ```

---

## 🩺 Clinical Disclaimer
*This deep learning application is designed for research, portfolio demonstration, and educational decision-support exploration only. All algorithmic predictions and heatmaps must be verified by certified radiologists prior to any clinical intervention.*
