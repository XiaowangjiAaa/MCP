import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.neighbors import KDTree
from typing import Tuple
from skimage.morphology import thin, remove_small_objects


def extract_skeleton_and_normals(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the skeleton of a crack mask and return placeholder normals.

    This simplified routine follows the provided script and performs
    ``thin`` and ``remove_small_objects`` cleanup.

    Args:
        mask: Binary mask where the crack is 1 and background is 0.

    Returns:
        skeleton_mask: binary skeleton image (0/1).
        skeleton_points: (N, 2) array of skeleton coordinates [x, y].
        normals: zeros placeholder kept for API compatibility.
    """
    # Step 1: extract the skeleton
    skeleton = skeletonize(mask > 0)

    # Step 2: optional refinement using ``thin``
    skeleton = thin(skeleton)

    # Step 3: remove small artifacts
    skeleton = remove_small_objects(skeleton, min_size=1)

    # Step 4: obtain skeleton points
    ys, xs = np.where(skeleton > 0)
    skeleton_points = np.stack([xs, ys], axis=1)

    # Step 5: return values
    skeleton_mask = skeleton.astype(np.uint8)
    normals = np.zeros_like(skeleton_points, dtype=np.float32)
    return skeleton_mask, skeleton_points, normals
