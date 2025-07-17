import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from MCP.quantify import quantify_crack_geometry
from MCP.tool import tool

@tool(name="visualize_crack_result")
def visualize_crack_result(subject_name: str, memory, visual_types: list = None, show: bool = True):
    """
    Dynamically display requested visualization layers, first trying paths
    stored in memory and falling back to defaults or on-the-fly generation
    (e.g. skeleton or max width).
    """

    
    if not visual_types:
        print("⚠️ visual_types not specified; nothing to show.")
        return {}

    images = {}
    fallback_needed = []


    for vtype in visual_types:
        path = None

        # try to fetch path from memory
        if vtype in {"mask", "skeleton", "max_width"}:

            path = memory.get_visualization_path(subject_name, vtype) if vtype != "mask" else memory.get_mask_path(subject_name)
            print(f"[🧪 DEBUG] vtype={vtype}, raw path from memory = {path}, type = {type(path)}")
            if not isinstance(path, (str, os.PathLike)):
                print(f"[❌ Type error] invalid path type for {vtype}: {type(path)} -> {path}")
                path = None
            path = Path(path) if path else None

        elif vtype == "original":
            path = Path(f"data/Test_images/{subject_name}.jpg")
        
        elif vtype == "ground_truth":
            path = Path(f"data/Test_images_GT/{subject_name}.png")
        
        # if not found in memory, try default paths
        if not path or not path.exists():
            if vtype == "mask":
                path = Path(f"outputs/masks/{subject_name}.png")
            elif vtype == "skeleton":
                path = Path(f"outputs/visuals/{subject_name}_skeleton.png")
            elif vtype == "max_width":
                path = Path(f"outputs/visuals/{subject_name}_max_width.png")

        # mark fallback if still missing
        if not path or not path.exists():
            if vtype in {"skeleton", "max_width"}:
                fallback_needed.append(vtype)
                continue
            else:
                print(f"[⚠️ Missing] path for {vtype} not found")
                continue

        # load image
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[vtype] = img
            memory.update_visualization_path(subject_name, vtype, str(path))  # write back to memory
        else:
            print(f"[❌ Error] unable to read image file: {path}")

    # === fallback: automatically generate skeleton / max_width etc ===
    if fallback_needed:
        mask_path = memory.get_mask_path(subject_name) or f"outputs/masks/{subject_name}.png"
        pixel_size = memory.get_pixel_size(subject_name) or 0.5

        if mask_path and os.path.exists(mask_path):
            print(f"[⚙️ fallback] generating layers: {fallback_needed}")
            result = quantify_crack_geometry(
                mask_path=mask_path,
                pixel_size_mm=pixel_size,
                metrics=[],
                visuals=fallback_needed
            )
            if result.get("status") == "success":
                for vtype in fallback_needed:
                    gen_path = result["visualizations"].get(vtype)
                    if gen_path and os.path.exists(gen_path):
                        img = cv2.imread(gen_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images[vtype] = img
                            memory.update_visualization_path(subject_name, vtype, gen_path)
        else:
            print(f"[❌ fallback] mask not found; cannot generate: {fallback_needed}")

    # === display images ===
    if not images:
        print("❌ no visualization layers to display")
        return {}

    if show:
        n = len(images)
        fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axs = [axs]
        for ax, (vtype, img) in zip(axs, images.items()):
            ax.imshow(img)
            ax.set_title(vtype.replace("_", " ").title())
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return {k: f"{subject_name}_{k}.png" for k in images.keys()}
