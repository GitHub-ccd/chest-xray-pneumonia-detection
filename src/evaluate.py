import os
import sys
import argparse

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from src.dataset import PneumoniaDataset, get_valid_transforms
from src.model import build_model
from src.gradcam import PyTorchGradCAM, overlay_cam_on_image

def evaluate_and_visualize(model_name="convnext", weights_path=None, data_dir="data", output_dir="images"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if weights_path is None:
        weights_path = f"src/best_{model_name}.pth"
        if not os.path.exists(weights_path):
            weights_path = "src/best_weights.pth"

    model = build_model(model_name=model_name, num_classes=2, pretrained=False).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights path '{weights_path}' not found. Using randomly initialized weights.")

    test_dataset = PneumoniaDataset(data_dir, split="test", transform=get_valid_transforms(224))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=["NORMAL", "PNEUMONIA"], digits=4)
    auc = roc_auc_score(all_targets, all_probs)

    print("\n================ TEST EVALUATION REPORT ================")
    print(f"Model Architecture: {model_name.upper()}")
    print(f"Accuracy: {acc*100:.2f}% | ROC-AUC: {auc*100:.2f}%\n")
    print(report)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_save_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_save_path}")

    # Generate Sample Grad-CAM Attention Visualizations
    print("\nGenerating sample Grad-CAM heatmap overlays...")
    gradcam = PyTorchGradCAM(model)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    normal_count = 0
    pneumonia_count = 0

    for idx in range(len(test_dataset)):
        img_tensor, target = test_dataset[idx]
        if (target == 0 and normal_count >= 4) or (target == 1 and pneumonia_count >= 4):
            continue

        input_batch = img_tensor.unsqueeze(0).to(device)
        cam, pred_class, prob = gradcam.generate_cam(input_batch)

        # Denormalize image for display
        rgb_img = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)

        overlay, _ = overlay_cam_on_image(rgb_img, cam, alpha=0.5)

        row = 0 if target == 0 else 1
        col = normal_count if target == 0 else pneumonia_count

        axes[row, col].imshow(overlay)
        true_label = "NORMAL" if target == 0 else "PNEUMONIA"
        pred_label = "NORMAL" if pred_class == 0 else "PNEUMONIA"
        axes[row, col].set_title(f"True: {true_label}\nPred: {pred_label} ({prob*100:.1f}%)",
                                 color="green" if true_label == pred_label else "red", fontsize=10)
        axes[row, col].axis('off')

        if target == 0:
            normal_count += 1
        else:
            pneumonia_count += 1

        if normal_count >= 4 and pneumonia_count >= 4:
            break

    plt.suptitle(f"Grad-CAM Diagnostic Attention Maps ({model_name.upper()})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    gradcam_save_path = os.path.join(output_dir, f"gradcam_samples_{model_name}.png")
    plt.savefig(gradcam_save_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM sample visualizations to {gradcam_save_path}")

    return {
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm,
        "report": report
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="convnext", choices=["convnext", "vit", "resnet", "baseline"])
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="images")
    args = parser.parse_args()

    evaluate_and_visualize(args.model_name, args.weights_path, args.data_dir, args.output_dir)
