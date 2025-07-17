# agent/nlp_parser.py

from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_image_indices_with_gpt(text: str) -> list:
    """Use GPT to extract image indices (0-based) from natural language.

    Input: natural language instruction
    Output: list of indices such as [0, 2, 4]
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
        print("⚠️ unable to parse GPT returned indices:", content)
        return []
    
def parse_visual_types_from_text(user_input: str) -> list:
    """Parse which visualization layers the user wants to display.

    Returns for example: ["original", "mask", "max_width"]
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
