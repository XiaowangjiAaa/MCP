import os
import cv2
import matplotlib.pyplot as plt
from typing import Optional


def visualize_result(
    image_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    overlay: bool = True,
    max_width_path: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Flexible visualization utility supporting combinations of original,
    segmentation and width maps.

    Args:
        image_path: path to the original image
        mask_path: segmentation mask path
        overlay: whether to overlay the mask on the original
        max_width_path: path to maximum width visualization (optional)
        save_path: if provided, save the figure; otherwise display
        title: title of the figure
    """
    if not image_path and not mask_path and not max_width_path:
        raise ValueError("At least one image path must be provided")

    visual_items = []

    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visual_items.append(("original", image))

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if overlay and image_path:
            overlay_img = image.copy()
            overlay_img[mask > 0] = [255, 0, 0]  # red overlay
            visual_items.append(("overlay", overlay_img))
        elif not overlay:
            visual_items.append(("mask", mask))

    if max_width_path:
        width_vis = cv2.imread(max_width_path)
        width_vis = cv2.cvtColor(width_vis, cv2.COLOR_BGR2RGB)
        visual_items.append(("max width", width_vis))

    n = len(visual_items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (sub_title, img) in zip(axes, visual_items):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(sub_title)
        ax.axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ visualization saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
