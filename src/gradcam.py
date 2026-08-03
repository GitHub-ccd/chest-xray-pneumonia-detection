import torch
import torch.nn.functional as F
import numpy as np
import cv2

class PyTorchGradCAM:
    """
    Robust PyTorch Grad-CAM implementation supporting CNNs and Vision Transformers.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        if self.target_layer is None:
            if hasattr(model, 'get_target_layer_for_gradcam'):
                layers = model.get_target_layer_for_gradcam()
                self.target_layer = layers[0] if layers else None
            else:
                # Find last Conv2d layer
                for module in reversed(list(model.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        self.target_layer = module
                        break

        if self.target_layer is not None:
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_category=None):
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_category is None:
            target_category = torch.argmax(probs, dim=1).item()

        score = output[0, target_category]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform attention map
            cam = np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
            return cam, target_category, probs[0, target_category].item()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        if len(gradients.shape) == 3: # (C, H, W)
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        elif len(gradients.shape) == 2: # ViT sequence output (N, C)
            weights = np.mean(gradients, axis=0)
            cam = np.sum(weights[:, None] * activations, axis=0)
            side = int(np.sqrt(cam.shape[0]))
            if side * side == cam.shape[0]:
                cam = cam.reshape(side, side)
            else:
                cam = cam.reshape(1, -1)

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam, target_category, probs[0, target_category].item()

def overlay_cam_on_image(rgb_image, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays normalized CAM mask (0-1) onto RGB image (0-255 uint8).
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if rgb_image.max() <= 1.0:
        rgb_image = np.uint8(255 * rgb_image)

    overlay = np.float32(heatmap) * alpha + np.float32(rgb_image) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay, heatmap
