from pathlib import Path
from typing import List, Tuple
import re

def list_image_paths(folder: str, suffixes: List[str] = [".jpg", ".png", ".jpeg"]) -> List[str]:
    """Generic function returning image paths within a folder (POSIX style)."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths = []
    for suf in suffixes:
        paths.extend(folder_path.glob(f"*{suf}"))

    # ensure stable ordering by filename
    return sorted([str(p.as_posix()) for p in paths], key=lambda x: Path(x).name)

def get_test_image_paths() -> List[str]:
    """Return all test image paths for segmentation (from data/Test_images)."""
    return list_image_paths("data/Test_images")

def get_test_image_by_index(index: int) -> str:
    """Return the image path at position ``index`` from Test_images.
    Supports ``index=-1`` for the last image."""
    images = get_test_image_paths()
    if index < 0:
        index = len(images) + index  # support -1 for last image
    if index < 0 or index >= len(images):
        raise IndexError(f"Image index {index} out of range; only {len(images)} images")
    return images[index]

def generate_segment_plan_from_paths(image_paths: list) -> list:
    return [
        {"tool": "segment_crack_image", "args": {"image_path": path}}
        for path in image_paths
    ]

def get_comparison_image_pairs() -> List[Tuple[str, str]]:
    """Pair GT and predicted mask paths from data/Test_images_GT and outputs/masks.
    Returns a list of tuples ``(gt_path, pred_path)``."""
    gt_paths = list_image_paths("data/Test_images_GT", suffixes=[".png"])
    pred_paths = list_image_paths("outputs/masks", suffixes=[".png"])

    # pair by file name
    gt_dict = {Path(p).name: p for p in gt_paths}
    pred_dict = {Path(p).name: p for p in pred_paths}

    pairs = []
    for name in gt_dict:
        if name in pred_dict:
            pairs.append((gt_dict[name], pred_dict[name]))

    return pairs

def get_csv_paths(results_dir: str = "outputs/csv/") -> Tuple[str, str]:
    """Return paths of prediction.csv and ground_truth.csv for comparison.
    Default directory is ``outputs/results``."""
    base = Path(results_dir)
    pred = base / "prediction.csv"
    gt = base / "ground_truth.csv"
    if not pred.exists() or not gt.exists():
        raise FileNotFoundError("prediction.csv or ground_truth.csv not found")

    return str(gt.as_posix()), str(pred.as_posix())

def extract_image_indices(text: str) -> List[int]:
    """Extract indices from natural language, e.g. "first image" or "image 3"."""
    text = text.lower()
    indices = []

    index_map = {
        "第一": 0, "第1": 0, "1st": 0, "image 1": 0, "first": 0,
        "第二": 1, "第2": 1, "2nd": 1, "image 2": 1, "second": 1,
        "第三": 2, "第3": 2, "3rd": 2, "image 3": 2, "third": 2,
    }
    for k, v in index_map.items():
        if k in text:
            indices.append(v)

    # regex match: image N
    matches = re.findall(r'image\\s*(\\d+)', text)
    indices.extend([int(m) - 1 for m in matches])

    return sorted(set(indices))
