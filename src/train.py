import os
import sys
import argparse
import time

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.dataset import PneumoniaDataset, get_train_transforms, get_valid_transforms
from src.model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return val_loss, val_acc, precision, recall, f1, auc

def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection PyTorch Model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-name", type=str, default="convnext", choices=["convnext", "vit", "resnet", "baseline"], help="Model backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--save-dir", type=str, default="src", help="Directory to save weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for training backbone: '{args.model_name}'")

    train_dataset = PneumoniaDataset(args.data_dir, split="train", transform=get_train_transforms(args.img_size))
    val_dataset = PneumoniaDataset(args.data_dir, split="val", transform=get_valid_transforms(args.img_size))
    test_dataset = PneumoniaDataset(args.data_dir, split="test", transform=get_valid_transforms(args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weighting to handle imbalance (Normal: 1341, Pneumonia: 3875)
    class_weights = torch.tensor([3875 / 1341, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(model_name=args.model_name, num_classes=2, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    save_path = os.path.join(args.save_dir, f"best_{args.model_name}.pth")

    print(f"\n--- Starting Training ({args.model_name}) ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Val F1: {f1*100:.2f}% | Val AUC: {auc*100:.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            # Also save general best_weights.pth
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_weights.pth"))
            print(f"  --> Saved new best checkpoint to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} mins.")

    # Evaluate on Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate(model, test_loader, criterion, device)
    print(f"TEST RESULTS ({args.model_name.upper()}):")
    print(f"  Accuracy  : {test_acc*100:.2f}%")
    print(f"  Precision : {test_prec*100:.2f}%")
    print(f"  Recall    : {test_rec*100:.2f}%")
    print(f"  F1-Score  : {test_f1*100:.2f}%")
    print(f"  ROC-AUC   : {test_auc*100:.2f}%")

if __name__ == "__main__":
    main()
