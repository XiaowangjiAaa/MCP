# agent/nlp_parser.py

from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_image_indices_with_gpt(text: str) -> list:
    """
    使用 GPT 模型从自然语言中提取用户想要处理的图像索引（从0开始）
    输入: 自然语言指令
    输出: 索引列表，例如 [0, 2, 4]
    """
    prompt = (
        "You are an assistant that extracts image indices from user instructions.\n"
        "Images are numbered from 0 (the first image), 1 (second image), etc.\n"
        "Return only a JSON list of integers such as [0, 2, 4] — no explanation, no text.\n\n"
        f"Instruction:\n{text}\n\nJSON output:"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You extract image indices from user instructions."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(json)?", "", content)
    content = re.sub(r"```$", "", content)

    try:
        return json.loads(content)
    except Exception:
        print("⚠️ 无法解析 GPT 返回的索引内容:", content)
        return []
    
def parse_visual_types_from_text(user_input: str) -> list:
    """
    从自然语言中解析用户希望展示的视觉图层类型。
    返回值示例：["original", "mask", "max_width"]
    """
    user_input = user_input.lower()
    types = []

    if "original" in user_input:
        types.append("original")
    if "ground truth" in user_input or "gt" in user_input:
        types.append("ground_truth")
    if "mask" in user_input or "prediction" in user_input:
        types.append("mask")
    if "skeleton" in user_input:
        types.append("skeleton")
    if "max width" in user_input or "width map" in user_input:
        types.append("max_width")
    if "normal" in user_input or "normals" in user_input:
        types.append("normals")

    return types
