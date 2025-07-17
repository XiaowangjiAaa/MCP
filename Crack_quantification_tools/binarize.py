import cv2
import numpy as np

def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Binarize an image to a 0/1 mask.

    The input may be color or grayscale; output is a uint8 array of 0/1.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Auto adapt: convert float images (0~1) to 0~255
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    _, binary = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)
