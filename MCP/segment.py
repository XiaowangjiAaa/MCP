import os
import torch
import numpy as np
import cv2
import traceback
from PIL import Image as PILImage
from torchvision import transforms
from MCP.tool import tool

from model.unet import UNet
from tools.preprocess import resolve_output_path

# Initialize model (load only once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def load_model(checkpoint_path: str = "checkpoints/unet_best.pth") -> torch.nn.Module:
    global _model
    if _model is None:
        model = UNet(in_channels=3, num_classes=1)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        _model = model
    return _model


@tool(name="segment_crack_image")
def segment_crack_image(image_path: str, checkpoint_path: str = "checkpoints/unet_best.pth") -> dict:
    """Image segmentation tool.

    Given an image path, output the mask image path (single channel 0/255).
    """
    try:
        # 1. read original image (BGR)
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 2. convert to PIL RGB and preprocess
        pil_img = PILImage.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((896, 896)),
            transforms.ToTensor()
        ])
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # 3. inference
        model = load_model(checkpoint_path)
        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = pred[0, 0].cpu().numpy()
            binary_mask = (pred_mask > 0.9).astype(np.uint8)

        # 4. save mask image
        output_path = resolve_output_path(image_path, suffix="mask", output_dir="outputs/masks")
        cv2.imwrite(output_path, binary_mask * 255)

        if not os.path.exists(output_path):
            raise RuntimeError(f"Mask save failed, file not found: {output_path}")

        return {
            "status": "success",
            "summary": "Segmentation complete; mask saved",
            "outputs": {
                "mask_path": output_path
            },
            "error": None
        }

    except Exception as e:
        print("[ERROR] segment_crack_image exception:", str(e))
        traceback.print_exc()
        return {
            "status": "error",
            "summary": "Segmentation failed",
            "outputs": None,
            "error": str(e)
        }
