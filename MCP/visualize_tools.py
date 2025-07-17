import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from MCP.quantify import quantify_crack_geometry
from MCP.tool import tool

@tool(name="visualize_crack_result")
def visualize_crack_result(subject_name: str, memory, visual_types: list = None, show: bool = True):
    """
    动态显示用户请求的视觉图层，优先从 memory 获取路径，
    若缺失则 fallback 至默认路径或生成（如 skeleton / max_width）。
    """

    
    if not visual_types:
        print("⚠️ 未指定 visual_types，不展示任何图层。")
        return {}

    images = {}
    fallback_needed = []


    for vtype in visual_types:
        path = None

        # ✅ 尝试从记忆中查路径
        if vtype in {"mask", "skeleton", "max_width"}:

            path = memory.get_visualization_path(subject_name, vtype) if vtype != "mask" else memory.get_mask_path(subject_name)
            print(f"[🧪 DEBUG] vtype={vtype}, raw path from memory = {path}, type = {type(path)}")
            if not isinstance(path, (str, os.PathLike)):
                print(f"[❌ 类型错误] {vtype} 路径类型不合法: {type(path)} -> 值: {path}")
                path = None
            path = Path(path) if path else None

        elif vtype == "original":
            path = Path(f"data/Test_images/{subject_name}.jpg")
        
        elif vtype == "ground_truth":
            path = Path(f"data/Test_images_GT/{subject_name}.png")
        
        # ✅ 如果记忆中无命中，尝试默认路径
        if not path or not path.exists():
            if vtype == "mask":
                path = Path(f"outputs/masks/{subject_name}.png")
            elif vtype == "skeleton":
                path = Path(f"outputs/visuals/{subject_name}_skeleton.png")
            elif vtype == "max_width":
                path = Path(f"outputs/visuals/{subject_name}_max_width.png")

        # ✅ 若仍无效，标记 fallback
        if not path or not path.exists():
            if vtype in {"skeleton", "max_width"}:
                fallback_needed.append(vtype)
                continue
            else:
                print(f"[⚠️ 缺失] {vtype} 图层找不到对应路径")
                continue

        # ✅ 读取图像
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[vtype] = img
            memory.update_visualization_path(subject_name, vtype, str(path))  # 写回记忆
        else:
            print(f"[❌ 错误] 无法读取图像文件: {path}")

    # === fallback: 自动生成 skeleton / max_width 等 ===
    if fallback_needed:
        mask_path = memory.get_mask_path(subject_name) or f"outputs/masks/{subject_name}.png"
        pixel_size = memory.get_pixel_size(subject_name) or 0.5

        if mask_path and os.path.exists(mask_path):
            print(f"[⚙️ fallback] 自动生成图层: {fallback_needed}")
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
            print(f"[❌ fallback] 找不到掩膜，无法生成: {fallback_needed}")

    # === 显示图像 ===
    if not images:
        print("❌ 没有任何可视化图层可展示")
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
