from pathlib import Path
from tools.path_utils import get_test_image_paths

# === Image index and filename mapping utilities ===

def get_image_name_by_index(index: int) -> str:
    """Return the normalized image name (without suffix) by index.

    Example: index=0 -> "0_crack"
    """
    paths = get_test_image_paths()
    if index < 0 or index >= len(paths):
        raise IndexError(f"Image index {index} out of range; total {len(paths)} images")
    return Path(paths[index]).stem

def get_index_by_image_name(name: str) -> int:
    """Find the index of an image (without extension) in Test_images."""
    paths = get_test_image_paths()
    for i, path in enumerate(paths):
        if Path(path).stem == name:
            return i
    raise ValueError(f"Image name not found: {name}")
