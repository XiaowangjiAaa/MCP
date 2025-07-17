from typing import List, Dict
import os
from pathlib import Path
from tools.path_utils import (
    get_test_image_by_index,
    get_test_image_paths,
    list_image_paths
)
from agent.nlp_parser import parse_image_indices_with_gpt
from agent.nlp_parser import parse_visual_types_from_text


def generate_plan(user_input: str, memory: Dict = None) -> List[Dict]:
    """
    基于规则和自然语言分析生成工具调用计划。
    支持 segment、quantify、generate（原 save）、compare、visualize、chat 等任务识别。
    """
    user_input = user_input.lower()
    plan = []
    memory = memory or {}
    pixel_size = memory.get("pixel_size_mm", 0.5)

    # === SEGMENT ===
    if "segment" in user_input or "detect" in user_input:
        if "all" in user_input:
            image_paths = get_test_image_paths()
        else:
            indices = parse_image_indices_with_gpt(user_input)
            image_paths = []
            for idx in indices:
                try:
                    image_paths.append(get_test_image_by_index(idx))
                except IndexError:
                    continue
        for path in image_paths:
            plan.append({
                "tool": "segment_crack_image",
                "task": "segment",
                "action": "segment",
                "subject": Path(path).stem,
                "args": {"image_path": path}
            })

    # === QUANTIFY ===
    if any(word in user_input for word in ["quantify", "geometry", "width"]):
        mask_dir = "outputs/masks"
        mask_paths = list_image_paths(mask_dir, suffixes=[".png"])
        selected = []

        if "all" in user_input:
            selected = mask_paths
        else:
            indices = parse_image_indices_with_gpt(user_input)
            if indices:
                for idx in indices:
                    if idx < len(mask_paths):
                        selected.append(mask_paths[idx])

        for path in selected:
            subject = Path(path).stem
            plan.append({
                "tool": "quantify_crack_geometry",
                "task": "quantify",
                "action": "quantify",
                "subject": subject,
                "args": {
                    "mask_path": path,
                    "pixel_size_mm": pixel_size,
                    "metrics": ["length", "area", "max_width", "avg_width"]
                }
            })

    # === GENERATE VISUALS ===
    if any(word in user_input for word in ["skeleton", "max width image", "width map", "visualize", "save"]):
        mask_dir = "outputs/masks"
        mask_paths = list_image_paths(mask_dir, suffixes=[".png"])
        selected = []

        if "all" in user_input:
            selected = mask_paths
        else:
            indices = parse_image_indices_with_gpt(user_input)
            if indices:
                for idx in indices:
                    if idx < len(mask_paths):
                        selected.append(mask_paths[idx])

        for path in selected:
            subject = Path(path).stem
            plan.append({
                "tool": "quantify_crack_geometry",
                "task": "generate",
                "action": "generate",
                "subject": subject,
                "args": {
                    "mask_path": path,
                    "pixel_size_mm": pixel_size,
                    "visuals": ["skeleton", "max_width"]
                }
            })

    # === VISUALIZE ONLY ===
    if any(word in user_input.lower() for word in ["visualize", "show", "display", "plot"]):
        selected = []

        if "all" in user_input.lower():
            selected = [Path(p).stem for p in list_image_paths("data/Test_images", [".jpg", ".png"])]
        else:
            indices = parse_image_indices_with_gpt(user_input)
            for idx in indices:
                try:
                    name = get_test_image_by_index(idx, return_name_only=True)
                    selected.append(name)
                except Exception:
                    continue

        visual_types = parse_visual_types_from_text(user_input)

        if not visual_types:
            print("❌ 用户未指定需要展示的图层，visualize 跳过")
        else:
            for name in selected:
                plan.append({
                    "tool": "visualize_crack_result",
                    "task": "visualize",
                    "action": "visualize",
                    "subject": name,
                    "args": {
                        "subject_name": name,
                        "visual_types": visual_types,
                        "show": True
                    }
                })

    RAG_KEYWORDS = [
        "advice", "summary", "suggestion", "standard", "规范",
        "as3735", "as3600", "crack width", "what is", "how does", "why",
        "limitations", "definition", "explanation", "allowable", "规定", "explain"
    ]

    if any(kw.lower() in user_input.lower() for kw in RAG_KEYWORDS):
        plan.append({
            "tool": "rag_answer",
            "task": "chat",
            "action": "chat",  # ✅ 必须写成 chat，不能是 answer
            "subject": "knowledge_query",
            "args": {
                "query": user_input
            }
        })


# === 示例测试入口 ===
if __name__ == "__main__":
    examples = [
        "segment the first image",
        "quantify all crack images",
        "save the max width map of all crack images",
        "visualize the skeleton map of second crack",
        "give me a summary advice"
    ]

    for i, prompt in enumerate(examples):
        print(f"\n🧠 Test {i+1}: {prompt}")
        p = generate_plan(prompt, memory={"pixel_size_mm": 0.5})
        for step in p:
            print(f"→ Tool: {step['tool']} | Task: {step.get('task')} | Action: {step.get('action')} | Subject: {step.get('subject')} | Args: {step['args']}")
