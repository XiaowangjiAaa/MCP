import numpy as np
import os
import cv2
from Crack_quantification_tools.binarize import binarize
from Crack_quantification_tools.skeleton import extract_skeleton_and_normals
from scipy.spatial import cKDTree

def visualize_max_width(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Visualize the maximum width by drawing a red line on the image.

    Returns the image with the line and the width value in pixels.
    """
    binary = binarize(image)
    _, skeleton_points, _ = extract_skeleton_and_normals(binary)

    if len(skeleton_points) == 0:
        return image.copy(), 0.0

    # extract contour points
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return image.copy(), 0.0
    contour_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    tree = cKDTree(contour_pts)

    # search for the maximum width between skeleton points and contours
    max_dist = 0.0
    best_pair = None
    for pt in skeleton_points:
        dists, idxs = tree.query([pt], k=2)
        if len(idxs[0]) < 2:
            continue
        p1, p2 = contour_pts[idxs[0][0]], contour_pts[idxs[0][1]]
        dist = np.linalg.norm(p1 - p2)
        if dist > max_dist:
            max_dist = dist
            best_pair = (p1, p2)

    # draw the maximum width line
    vis = image.copy()
    if best_pair:
        p1, p2 = map(tuple, best_pair)
        p1, p2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
        cv2.line(vis, p1, p2, (0, 0, 255), 2)  # red line
        cv2.circle(vis, p1, 3, (0, 255, 0), -1)
        cv2.circle(vis, p2, 3, (0, 255, 0), -1)

    return vis, round(max_dist, 2)

def draw_skeleton_overlay(image: np.ndarray, centers: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay skeleton points on the original image in red."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    output = image.copy()
    for pt in centers:
        x, y = int(pt[1]), int(pt[0])  # note coordinate order (x, y)
        cv2.circle(output, (x, y), radius=1, color=(255, 0, 0), thickness=-1)
    return cv2.addWeighted(image, 1 - alpha, output, alpha, 0)


def save_visual(image: np.ndarray, save_path: str) -> str:
    """Save a visualization image to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    return save_path