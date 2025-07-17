import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "你是一个裂缝图像分析任务的智能规划器。\n"
    "你的任务是根据用户的自然语言请求，生成一个结构化任务计划，每个任务为一个 step，包含以下字段：\n"
    "\n"
    "- action: 必选，取值如下：segment / quantify / visualize / generate / compare / chat\n"
    "- target_indices: 必选，图像索引数组，如 [0] 表示第一张图像，或使用 'all'。\n"
    "- pixel_size_mm: 可选，仅在 quantify 和 generate 中使用，单位毫米。\n"
    "- metrics: 可选，仅 quantify 使用，如 max_width、area、length、avg_width。\n"
    "- visual_types: 可选，仅 visualize 和 generate 使用，如 skeleton、max_width、overlay、normals、mask、original。\n"
    "- tool: 可选，当 action 为 chat 且问题为法规/规范/建议类时，必须指定为 \"rag_answer\"。\n"
    "- args: 可选，仅 chat 动作使用，包含字段 query，用于传递用户的原始提问内容。\n"
    "\n"
    "🎯 各任务定义如下（彼此不重叠）：\n"
    "1. segment：执行图像分割，输出掩膜图。\n"
    "2. quantify：基于掩膜图，计算裂缝几何指标。需 pixel_size，可选 metrics。\n"
    "3. visualize：展示已有图像（如原图、掩膜、骨架、宽度图）。不生成文件，也不写入记录。必须指定 visual_types。\n"
    "4. generate：生成并保存图像（如骨架、法向图、宽度图等），需提供 pixel_size 与 visual_types，生成后可供 visualize 调用。\n"
    "5. compare：比较两个或多个分析结果（如误差、相似度等）。\n"
    "6. chat：用于两类情况：\n"
    "   ① 闲聊类请求 → 只写 action 和 target_indices 即可；\n"
    "   ② 涉及法规、规范、标准、定义或建议类问题（如包含关键词 regulation, advice, requirement, according to, AS3600, crack width 等）时，说明用户希望查询知识库，此时：\n"
    "      - 必须指定 tool 为 \"rag_answer\"\n"
    "      - 并在 args 中提供 query 字段，填写用户原始问题\n"
    "\n"
    "📌 编排规则（必须遵守）：\n"
    "- 仅当用户说“save, generate, export, create”等保存相关词汇时，才使用 generate。\n"
    "- 仅当用户说“show, visualize, display, plot”等查看相关词汇时，使用 visualize。\n"
    "- 严禁将 visualize 错误识别为 generate，即使用户提到 skeleton、width map 等关键词，也不得生成。\n"
    "- pixel_size 只用于 quantify 与 generate。\n"
    "- metrics 仅在 quantify 时填写，visualize / generate 不可填写。\n"
    "- visual_types 仅在 visualize 与 generate 使用。\n"
    "- 用户意图复合时，请拆分为多个 step。\n"
    "- chat 动作为法规问答时，必须包含 tool 与 args.query 字段。\n"
    "\n"
    "✅ 示例 1：\n"
    "用户说：“save the skeleton and max width of this crack image”\n"
    "输出：\n"
    "{\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"action\": \"generate\",\n"
    "      \"target_indices\": [0],\n"
    "      \"pixel_size_mm\": 0.5,\n"
    "      \"visual_types\": [\"skeleton\", \"max_width\"]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "✅ 示例 2：\n"
    "用户说：“visualize the max width and skeleton map of the first image”\n"
    "输出：\n"
    "{\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"action\": \"visualize\",\n"
    "      \"target_indices\": [0],\n"
    "      \"visual_types\": [\"max_width\", \"skeleton\"]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "✅ 示例 3：\n"
    "用户说：“segment and then quantify the second image”\n"
    "输出：\n"
    "{\n"
    "  \"steps\": [\n"
    "    { \"action\": \"segment\", \"target_indices\": [1] },\n"
    "    { \"action\": \"quantify\", \"target_indices\": [1], \"pixel_size_mm\": 0.5 }\n"
    "  ]\n"
    "}\n"
    "\n"
    "✅ 示例 4（法规问答）：\n"
    "用户说：“According to AS3600, what crack widths are allowed?”\n"
    "输出：\n"
    "{\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"action\": \"chat\",\n"
    "      \"target_indices\": [\"all\"],\n"
    "      \"tool\": \"rag_answer\",\n"
    "      \"args\": {\n"
    "        \"query\": \"According to AS3600, what crack widths are allowed?\"\n"
    "      }\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "请严格按照结构输出 JSON 对象，仅包含字段 steps，不要添加注释、解释或多余文字。"
)



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
    """
    使用 GPT function calling 输出多步骤分析计划。
    每个 step 是一个结构体，字段受控，提升稳定性。
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
        print("🧠 Planner 输出步骤：")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print("❌ 解析计划失败:", e)
        print("返回内容:", response.choices[0].message.content)
        return {"steps": []}
