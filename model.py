import torch
from unet import UNet


def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(device, model_path):
    model = UNet(in_channels=6, out_channels=1).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model
