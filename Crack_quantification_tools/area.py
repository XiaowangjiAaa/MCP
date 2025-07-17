import numpy as np

def compute_crack_area_px(mask: np.ndarray) -> int:
    """Compute crack area in pixels (number of white pixels).

    Supports both 0/1 and 0/255 binary images.
    """
    binary = (mask > 0).astype(np.uint8)
    return int(np.sum(binary))
