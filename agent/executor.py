from typing import List, Dict, Any
import os
import traceback
from MCP.tool import tool_registry
from agent.object_memory_manager import ObjectMemoryManager

object_store = ObjectMemoryManager()

def patch_image_paths(plan: list, base_folder: str = "data") -> list:
    for step in plan:
        args = step.get("args", {})
        for key in args:
            if "path" in key and isinstance(args[key], str):
                path = args[key]
                if not os.path.exists(path) and not os.path.dirname(path):
                    args[key] = os.path.join(base_folder, path)
    return plan

def execute_plan(plan: List[Dict[str, Any]], memory=None) -> List[Dict[str, Any]]:
    results = []

    for step in plan:
        tool_name = step.get("tool")
        args = step.get("args", {})
        subject = step.get("subject", "")
        action = step.get("action", "")  # ✅ 显式获取 action 字段

        tool_fn = tool_registry.get(tool_name)
        print("🧰 当前可用工具列表:", list(tool_registry.keys()))


        # ⚠️ 清理空 visuals
        if tool_name == "quantify_crack_geometry":
            visuals = args.get("visuals", None)
            if visuals is not None and len(visuals) == 0:
                del args["visuals"]

        # ✅ 工具未注册
        if not callable(tool_fn):
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"工具未注册: {tool_name}",
                "outputs": None,
                "visualizations": None,
                "error": "Tool not found in registry",
                "args": args,
                "subject": subject,
                "action": action
            })
            continue

        try:
            # ✅ 自动补充 pixel_size_mm
            if tool_name in ["quantify_crack_geometry", "generate_crack_visuals"]:
                if "pixel_size_mm" not in args or args["pixel_size_mm"] is None:
                    if memory is not None and hasattr(memory, "get_pixel_size"):
                        pixel_from_memory = memory.get_pixel_size(subject)
                        if pixel_from_memory is not None:
                            args["pixel_size_mm"] = pixel_from_memory
                            print(f"[🔁] 自动补充 pixel_size_mm={pixel_from_memory} for subject={subject}")

            # ✅ 执行工具函数
            result = tool_fn(**args)

            # ✅ 更新 object store
            if tool_name == "segment_crack_image" and result.get("status") == "success":
                image_path = args.get("image_path", "")
                object_id = object_store.find_id_by_image_path(image_path)
                mask_path = result["outputs"].get("mask_path")
                if object_id and mask_path:
                    object_store.update(object_id, "segmentation_path", mask_path)
                    object_store.add_status(object_id, "segmented")

            elif tool_name == "quantify_crack_geometry" and result.get("status") == "success":
                mask_path = args.get("mask_path", "")
                object_id = object_store.find_id_by_mask_path(mask_path)
                vis_path = result.get("visualizations", {}).get("max_width_overlay")
                if object_id:
                    if vis_path:
                        object_store.update(object_id, "visualization_path", vis_path)
                    object_store.add_status(object_id, "quantified")

            # ✅ 汇总结果
            outputs = result.get("outputs", {}) or {}
            visuals = result.get("visualizations", {}) or {}

            merged_outputs = {**outputs, **visuals}

            result_record = {
                "tool": tool_name,
                "status": result.get("status", "unknown"),
                "summary": result.get("summary", ""),
                "outputs": merged_outputs,
                "visualizations": visuals,
                "error": result.get("error", None),
                "args": args,
                "subject": subject,
                "action": action
            }

            results.append(result_record)

            # ✅ 写入 memory（根据 action 判断是否量化或生成）
            if memory is not None and hasattr(memory, "handle_result"):
                memory.handle_result(subject, tool_name, result, step)

        except Exception as e:
            print(f"[❌ ERROR] 工具 {tool_name} 执行时出错: {e}")
            traceback.print_exc()
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"执行 {tool_name} 失败",
                "outputs": None,
                "visualizations": None,
                "error": traceback.format_exc(),
                "args": args,
                "subject": subject,
                "action": action
            })

    return results
