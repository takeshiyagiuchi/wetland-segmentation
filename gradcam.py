import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from model import set_device

device = set_device()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)  # ✅ FIXED

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image):
        self.model.zero_grad()

        output = self.model(input_image)
        output = torch.sigmoid(output)

        loss = output.mean()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)

        cam = F.relu(cam)

        # remove batch dimension safely
        cam = cam.squeeze(0).detach().cpu().numpy()

        # resize to input size
        cam = cv2.resize(cam, (256, 256))

        # normalize safely
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def show_gradcam(model, img, mask, grad_cam, save_path=None):
    model.eval()

    # convert numpy → tensor
    input_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    # ---------- Prediction ----------
    with torch.no_grad():
        pred = model(input_img)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()

    # ---------- Grad-CAM ----------
    cam = grad_cam.generate(input_img)

    # ---------- Image ----------
    img_np = input_img.squeeze(0).cpu().numpy()

    # RGB from bands (B4, B3, B2 → [2,1,0])
    rgb = img_np[[2, 1, 0]]
    rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    # ---------- Plot ----------
    plt.figure(figsize=(16, 4))

    # Image
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("Image")

    # Ground truth
    plt.subplot(1, 4, 2)
    plt.imshow(mask[0], cmap="gray")
    plt.title("Ground Truth")

    # Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Prediction")

    # Grad-CAM
    plt.subplot(1, 4, 4)
    plt.imshow(rgb)
    heatmap = plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # ---------- Save ----------
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to: {save_path}")

    plt.show()

def get_target_layer(model, layer_name):
    return {
        "down1": model.down1,
        "down2": model.down2,
        "down3": model.down3,
        "down4": model.down4,
        "bottleneck": model.bottleneck,
        "conv4": model.conv4,
        "conv3": model.conv3,
        "conv2": model.conv2,
        "conv1": model.conv1,
    }[layer_name]
