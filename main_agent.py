import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agent.executor import execute_plan
from tools.path_utils import get_test_image_paths, get_test_image_by_index
from MCP.visualize_tools import visualize_crack_result
from agent.gpt_intent_parser import generate_composite_plan
from agent.object_memory_manager import ObjectMemoryManager
from agent.session_manager import SessionManager



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

session = SessionManager()
logger = session.get_logger()
memory = session.get_memory()
object_store = ObjectMemoryManager()

DEFAULT_METRICS = ["max_width", "avg_width", "length", "area"]


def generate_agent_reply(user_input: str, plan: dict, results: list) -> str:
    rag_result = next((r for r in results if r.get("tool") == "rag_answer" and r.get("status") == "success"), None)
    if rag_result:
        return rag_result["outputs"].get("answer", "")

    prompt = (
        "You are an AI assistant focused on crack image analysis.\n"
        "Summarize the task results using natural language.\n"
        "Keep the reply short and mention image indices, operations and key results when possible.\n\n"
        f"🖍️ User request: {user_input}\n"
        f"📋 Plan: {json.dumps(plan, ensure_ascii=False)}\n"
        f"✅ Results: {json.dumps(results, ensure_ascii=False)}\n\n"
        "Provide the answer only, no additional explanations or formatting."
    )
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a professional crack image analysis AI that returns concise summaries."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def chat_fallback(user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an AI agent focused on crack image analysis."},
            {"role": "user", "content": user_input}
        ]
    )
    reply = response.choices[0].message.content.strip()
    logger.log_agent(reply)
    return reply


def normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")


def match_metric_key(requested: str, candidate: str) -> bool:
    return normalize(requested) in normalize(candidate)


if __name__ == "__main__":
    while True:
        user_input = input("\n🧠 Enter a command (or exit): ")
        if user_input.strip().lower() in {"exit", "quit"}:
            session.export_memory_snapshot()
            session.print_summary()
            break

        logger.log_user(user_input)
        print("🗽 Understanding intent...")

        plan = generate_composite_plan(user_input)
        steps = plan.get("steps", [])

        if all(step["action"] == "chat" for step in steps):
            has_rag_tool = any(step.get("tool", "").lower() == "rag_answer" for step in steps)
            
            if has_rag_tool:
                print("📚 Executing rag_answer for knowledge query...")
                tool_plan = steps  # use planner-generated steps directly
                results = execute_plan(tool_plan, memory=memory)
                reply = generate_agent_reply(user_input, plan, results)
                print("\n💬 AI answer:")
                print(reply)
                logger.log_agent(reply)
                continue
            
            else:
                print("💬", chat_fallback(user_input))
                continue

        print("\n📋 Task plan:")
        for i, step in enumerate(steps):
            print(f"{i+1}. {step['action']} → index={step.get('target_indices')}")

        tool_plan = []
        results = []
        all_images = get_test_image_paths()
        index_to_image_name = {
            i: os.path.basename(p).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            for i, p in enumerate(all_images)
        }

        for step in steps:
            action = step["action"]
            indices = step.get("target_indices", [])

            if isinstance(indices, list) and "all" in indices:
                indices = list(range(len(all_images)))
            elif isinstance(indices, str) and indices == "all":
                indices = list(range(len(all_images)))

            step["target_indices"] = indices
            pixel_size = step.get("pixel_size_mm", 0.5)
            metrics = step.get("metrics", [])
            visual_types = step.get("visual_types", [])

            if action == "quantify" and not metrics:
                metrics = DEFAULT_METRICS
                step["metrics"] = metrics
            if action == "visualize" and not visual_types:
                visual_types = ["original", "mask"]
                step["visual_types"] = visual_types

            if not indices:
                print(f"⚠️ step [{action}] missing image indices")
                continue

            for i in indices:
                img_path = get_test_image_by_index(i)
                name = index_to_image_name[i]
                object_store.register_image(img_path)
                mask_name = os.path.basename(img_path).replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join("outputs/masks", mask_name)

                if action == "segment":
                    if os.path.exists(mask_path):
                        print(f"♻️ Mask already exists, skipping segment: {mask_path}")
                        results.append({
                            "tool": "segment_crack_image",
                            "status": "success",
                            "summary": "Mask already exists on disk, skipping",
                            "outputs": {"mask_path": mask_path},
                            "visualizations": None,
                            "args": {"image_path": img_path},
                            "subject": name
                        })
                        continue
                    tool_plan.append({
                        "tool": "segment_crack_image",
                        "args": {"image_path": img_path},
                        "subject": name
                    })

                elif action == "quantify":
                    if memory.has_metrics(name, metrics, pixel_size):
                        print(f"✅ Metrics for image {name} already in memory; using cached results.")
                        metric_values = memory.get_metrics_by_name(name, pixel_size)
                        outputs = {k: v for k, v in metric_values.items() if any(map(lambda m: m.lower() in k.lower(), metrics))}
                        results.append({
                            "tool": "quantify_crack_geometry",
                            "status": "success",
                            "summary": f"Loaded from memory with {len(outputs)} metrics",
                            "outputs": outputs,
                            "visualizations": None,
                            "args": {
                                "mask_path": mask_path,
                                "pixel_size_mm": pixel_size,
                                "metrics": metrics,
                                "visuals": []
                            },
                            "subject": name
                        })
                        continue

                    args = {
                        "mask_path": mask_path,
                        "pixel_size_mm": pixel_size,
                        "metrics": metrics
                    }
                    if "visual_types" in step:
                        args["visuals"] = step["visual_types"]

                    tool_plan.append({
                        "tool": "quantify_crack_geometry",
                        "args": args,
                        "subject": name
                    })

                elif action == "generate":
                    args = {
                        "mask_path": mask_path,
                        "pixel_size_mm": pixel_size,
                        "metrics": [],
                        "visuals": visual_types
                    }
                    tool_plan.append({
                        "tool": "quantify_crack_geometry",
                        "args": args,
                        "subject": name
                    })

                elif action == "visualize":
                    vis_paths = visualize_crack_result(subject_name=name, memory=memory, visual_types=visual_types)
                    results.append({
                        "tool": "visualize_crack_result",
                        "status": "success" if vis_paths else "no_output",
                        "summary": f"Generated {len(vis_paths)} visuals" if vis_paths else "No visualization produced",
                        "outputs": {},
                        "visualizations": vis_paths,
                        "args": {"subject_name": name, "visual_types": visual_types},
                        "subject": name
                    })

        if not tool_plan and results:
            print("♻️ All steps hit in memory; no tools executed.")
        else:
            print("\n🚀 Executing:")
            for step in tool_plan:
                print(f"→ {step['tool']}({step['args']})")
            exec_results = execute_plan(tool_plan, memory=memory)
            results += exec_results

            # save generated visualization paths to memory
            for r in exec_results:
                if r['tool'] == "quantify_crack_geometry" and r['status'] == "success":
                    subject = r["subject"]
                    visual_paths = r.get("visualizations", {})
                    if visual_paths:
                        memory.save_visualizations(subject, pixel_size, visual_paths)

        for r in results:
            print(f"[{r['tool']}] → {r['status']}")
            if r['tool'] == "quantify_crack_geometry" and r['status'] == "success":
                print("📊 Requested metrics:")
                for k, v in r.get("outputs", {}).items():
                    print(f"  {k}: {v}")

        for step in tool_plan:
            if "action" not in step:
                step["action"] = step.get("task", "")  # keep compatibility with memory logic

        memory.update_context(
            intent="multi_step",
            indices=[i for step in steps for i in step.get("target_indices", [])],
            pixel_size=pixel_size,
            results=results,
            plan=tool_plan
        )

        logger.log_agent_structured({
            "intent": "multi_step",
            "user_input": user_input,
            "steps": steps,
            "tool_plan": tool_plan,
            "result": results,
            "message": "✅ Multi-step task completed"
        })

        reply = generate_agent_reply(user_input, tool_plan, results)
        print("\n💬 AI summary:")
        print(reply)
        logger.log_agent(reply)
