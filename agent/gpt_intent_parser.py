import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are a planning assistant for crack image analysis tasks.\n"
    "Generate a structured task plan from the user's request. Each step must include:\n"
    "- action: one of segment / quantify / visualize / generate / compare / chat\n"
    "- target_indices: list of image indices such as [0] or 'all'\n"
    "- pixel_size_mm: optional, used only for quantify and generate (in millimeters)\n"
    "- metrics: optional, only for quantify (e.g. max_width, area, length, avg_width)\n"
    "- visual_types: optional, only for visualize or generate (skeleton, max_width, overlay, normals, mask, original)\n"
    "- tool: optional, must be 'rag_answer' when action is chat for regulation questions\n"
    "- args: optional, used for chat with field 'query' containing the original question\n"
    "\n"
    "Task definitions:\n"
    "1. segment: perform image segmentation to produce a mask.\n"
    "2. quantify: compute crack metrics from a mask; requires pixel_size_mm; metrics optional.\n"
    "3. visualize: display existing images such as original, mask, skeleton or width maps. No files are generated. visual_types must be specified.\n"
    "4. generate: create and save images (e.g. skeleton, normals, width maps). Requires pixel_size_mm and visual_types. Generated files can later be visualized.\n"
    "5. compare: compare two or more analysis results.\n"
    "6. chat: either casual conversation or knowledge-base questions. For regulation queries specify tool='rag_answer' and include args.query.\n"
    "\n"
    "Rules:\n"
    "- Use generate only when the user requests saving or creating images.\n"
    "- Use visualize only when the user requests to show or display images.\n"
    "- Never confuse visualize with generate even if keywords like skeleton or width map appear.\n"
    "- pixel_size_mm applies only to quantify and generate.\n"
    "- metrics field is only for quantify.\n"
    "- visual_types is only for visualize and generate.\n"
    "- Split multiple intents into separate steps.\n"
    "- For regulation Q&A include tool and args.query.\n"
    "\n"
    "Return only a JSON object with the 'steps' field and no extra text." )



FUNCTION_SCHEMA = [
    {
        "name": "generate_composite_plan",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["segment", "quantify", "visualize", "compare", "chat", "generate"]
                            },
                            "target_indices": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "integer"},
                                        {"type": "string", "enum": ["all"]}
                                    ]
                                }
                            },
                            "pixel_size_mm": {"type": "number"},
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["max_width", "avg_width", "length", "area"]
                                }
                            },
                            "visual_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["original", "mask", "overlay", "max_width", "skeleton", "normals"]
                                }
                            }
                        },
                        "required": ["action", "target_indices"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
]

def generate_composite_plan(user_input: str) -> dict:
    """Use GPT function calling to output a multi-step analysis plan.
    Each step is a structured object for stability.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        functions=FUNCTION_SCHEMA,
        function_call={"name": "generate_composite_plan"}
    )

    try:
        result = json.loads(response.choices[0].message.function_call.arguments)
        print("🧠 Planner output steps:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print("❌ Failed to parse plan:", e)
        print("Returned content:", response.choices[0].message.content)
        return {"steps": []}
