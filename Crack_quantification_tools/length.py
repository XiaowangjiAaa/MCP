import numpy as np
from .skeleton import extract_skeleton_and_normals

def compute_crack_length_px(mask: np.ndarray) -> int:
    """Estimate crack length in pixels using the number of skeleton points."""
    _, skeleton_points, _ = extract_skeleton_and_normals(mask)
    return len(skeleton_points)
