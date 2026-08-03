import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

class CustomBaselineCNN(nn.Module):
    """
    Standard baseline CNN architecture for binary chest X-ray classification.
    """
    def __init__(self, num_classes=2):
        super(CustomBaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PneumoniaModel(nn.Module):
    """
    Pneumonia Detection Model supporting ConvNeXt, ViT, and Custom CNN backbones.
    """
    def __init__(self, backbone_name="convnext", num_classes=2, pretrained=True):
        super(PneumoniaModel, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "convnext":
            self.model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
        elif self.backbone_name == "vit":
            self.model = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
        elif self.backbone_name == "resnet":
            self.model = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
        elif self.backbone_name == "baseline":
            self.model = CustomBaselineCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from ['convnext', 'vit', 'resnet', 'baseline'].")

    def forward(self, x):
        return self.model(x)

    def get_target_layer_for_gradcam(self):
        """
        Returns target layer for Grad-CAM activation visualization.
        """
        if self.backbone_name == "convnext":
            # For convnext_tiny in timm, stages[-1].blocks[-1] or stages[-1]
            return [self.model.stages[-1].blocks[-1]]
        elif self.backbone_name == "vit":
            # For DeiT/ViT in timm, block norm before head
            return [self.model.blocks[-1].norm1]
        elif self.backbone_name == "resnet":
            return [self.model.layer4[-1]]
        elif self.backbone_name == "baseline":
            return [self.model.features[-2]] # Conv2d(128, 256)
        return None

def build_model(model_name="convnext", num_classes=2, pretrained=True):
    return PneumoniaModel(backbone_name=model_name, num_classes=num_classes, pretrained=pretrained)
