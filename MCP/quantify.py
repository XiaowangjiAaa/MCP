import os
import cv2
import numpy as np
import traceback
from MCP.tool import tool

from Crack_quantification_tools.binarize import binarize
from Crack_quantification_tools.area import compute_crack_area_px
from Crack_quantification_tools.length import compute_crack_length_px
from Crack_quantification_tools.width_max import compute_max_width_px
from Crack_quantification_tools.width_avg import compute_average_width_px
from Crack_quantification_tools.skeleton import extract_skeleton_and_normals
from tools.visualize import visualize_max_width, save_visual
from tools.io_utils import append_to_csv


@tool(name="quantify_crack_geometry")
def quantify_crack_geometry(
    mask_path: str,
    pixel_size_mm: float,
    metrics: list = None,
    visuals: list = None
) -> dict:
    """Crack image quantification tool with optional visualization.

    Selected metrics are computed and written to CSV.
    """
    try:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # original image for visualization
        img_raw = cv2.imread(mask_path)
        if img_raw is None:
            raise ValueError(f"Invalid image format: {mask_path}")

        # binarize image for analysis
        mask = binarize(img_raw)

        # compute metrics
        all_metrics = {
            "Length (mm)": lambda: round(compute_crack_length_px(mask) * pixel_size_mm, 2),
            "Area (mm^2)": lambda: round(compute_crack_area_px(mask) * pixel_size_mm ** 2, 2),
            "Max Width (mm)": lambda: round(compute_max_width_px(mask) * pixel_size_mm, 2),
            "Avg Width (mm)": lambda: round(compute_average_width_px(mask) * pixel_size_mm, 2),
        }

        metric_alias_map = {
            "Length (mm)": "length",
            "Area (mm^2)": "area",
            "Max Width (mm)": "max_width",
            "Avg Width (mm)": "avg_width"
        }

        if not metrics:
            selected_metrics = list(all_metrics.keys())
        else:
            selected_metrics = []
            for m in metrics:
                for k in all_metrics:
                    if m.lower().replace(" ", "").replace("_", "") in k.lower().replace(" ", "").replace("_", ""):
                        selected_metrics.append(k)

        print(f"\U0001F4D0 selected metrics: {selected_metrics}")

        results = {
            metric_alias_map[name]: all_metrics[name]() for name in selected_metrics
        }

        # write to CSV (keep original column names with units)
        results_for_csv = {name: all_metrics[name]() for name in selected_metrics}
        image_name = os.path.splitext(os.path.basename(mask_path))[0]
        os.makedirs("outputs/csv", exist_ok=True)
        append_to_csv("outputs/csv/predicted_metrics.csv", image_name, results_for_csv)

        # visualization results
        vis_results = {}
        if visuals:
            image_base = os.path.splitext(os.path.basename(mask_path))[0]
            visual_dir = os.path.join("outputs", "visuals")
            os.makedirs(visual_dir, exist_ok=True)

            # skeleton and normals
            _, centers, normals = extract_skeleton_and_normals(mask)

            # 1. skeleton points (red)
            if "skeleton" in visuals or "all" in visuals:
                overlay = img_raw.copy()
                for pt in centers:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(overlay, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
                path = save_visual(overlay, os.path.join(visual_dir, f"{image_base}_skeleton.png"))
                vis_results["skeleton_overlay"] = path

            # 2. normal vectors (green lines)
            if "normals" in visuals or "all" in visuals:
                normal_overlay = img_raw.copy()
                for pt, n in zip(centers, normals):
                    x, y = int(pt[0]), int(pt[1])
                    dx, dy = n[0], n[1]
                    pt2 = (int(x + dx * 10), int(y + dy * 10))
                    cv2.arrowedLine(normal_overlay, (x, y), pt2, (0, 255, 0), 1, tipLength=0.3)
                path = save_visual(normal_overlay, os.path.join(visual_dir, f"{image_base}_skeleton_normals.png"))
                vis_results["skeleton_normals"] = path

            # 3. maximum width line (overlaid on original image)
            if "max_width" in visuals or "all" in visuals:
                vis_width, max_width = visualize_max_width(img_raw)
                path = save_visual(vis_width, os.path.join(visual_dir, f"{image_base}_max_width.png"))
                vis_results["max_width_overlay"] = path

        return {
            "status": "success",
            "summary": f"Quantification complete: {len(results)} metrics, {len(vis_results)} visuals",
            "outputs": results,
            "visualizations": vis_results,
            "error": None
        }

    except Exception as e:
        try:
            print("[❗] quantify_crack_geometry caught exception:", str(e))
            traceback.print_exc()
        except Exception as log_error:
            print("[❗] ERROR printing traceback:", str(log_error))
        return {
            "status": "error",
            "summary": "Quantification failed",
            "outputs": None,
            "visualizations": None,
            "error": str(e)
        }