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
        action = step.get("action", "")  # explicitly get the action field

        tool_fn = tool_registry.get(tool_name)
        print("🧰 available tools:", list(tool_registry.keys()))


        # remove empty visuals field
        if tool_name == "quantify_crack_geometry":
            visuals = args.get("visuals", None)
            if visuals is not None and len(visuals) == 0:
                del args["visuals"]

        # tool not registered
        if not callable(tool_fn):
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"Tool not registered: {tool_name}",
                "outputs": None,
                "visualizations": None,
                "error": "Tool not found in registry",
                "args": args,
                "subject": subject,
                "action": action
            })
            continue

        try:
            # auto fill pixel_size_mm
            if tool_name in ["quantify_crack_geometry", "generate_crack_visuals"]:
                if "pixel_size_mm" not in args or args["pixel_size_mm"] is None:
                    if memory is not None and hasattr(memory, "get_pixel_size"):
                        pixel_from_memory = memory.get_pixel_size(subject)
                        if pixel_from_memory is not None:
                            args["pixel_size_mm"] = pixel_from_memory
                            print(f"[🔁] auto-filled pixel_size_mm={pixel_from_memory} for subject={subject}")

            # execute tool function
            result = tool_fn(**args)

            # update object store
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

            # compose result record
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

            # write to memory based on action
            if memory is not None and hasattr(memory, "handle_result"):
                memory.handle_result(subject, tool_name, result, step)

        except Exception as e:
            print(f"[❌ ERROR] tool {tool_name} failed: {e}")
            traceback.print_exc()
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"Execution of {tool_name} failed",
                "outputs": None,
                "visualizations": None,
                "error": traceback.format_exc(),
                "args": args,
                "subject": subject,
                "action": action
            })

    return results
