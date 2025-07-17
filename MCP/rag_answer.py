import re
from typing import List, Optional
from langdetect import detect
from openai import OpenAI
from RAG_knowledge.retriever import retrieve_context
from MCP.tool import tool

client = OpenAI()


def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return "zh" if "zh" in lang else "en"
    except:
        return "zh"  # 默认中文


def format_context(contexts: List[dict], lang: str) -> str:
    text_key = "text_zh" if lang == "zh" else "text_en"
    section_key = "section_zh" if lang == "zh" else "section_en"

    lines = []
    for i, chunk in enumerate(contexts, 1):
        source = chunk.get("source", "未知")
        section = chunk.get(section_key, "N/A")
        text = chunk.get(text_key, "").strip()
        lines.append(f"[{i}] 来源: {source} | 章节: {section}\n{text}")
    return "\n\n".join(lines)


@tool(name="rag_answer")
def rag_answer(query: str, lang: Optional[str] = "auto", top_k: int = 5, model: str = "gpt-4") -> dict:
    # 自动检测语言
    detected_lang = detect_lang(query) if lang == "auto" else lang

    # 检索段落
    contexts = retrieve_context(query, lang=detected_lang, top_k=top_k)

    # ⚠️ 打印检索内容（调试用）
    print("\n====== [🔍 RAG 检索到的段落] ======")
    print(format_context(contexts, detected_lang))
    print("====================================\n")

    # 构建 prompt
    context_text = format_context(contexts, detected_lang)
    if detected_lang == "zh":
        prompt = f"""
你是一位专业结构工程助手，下面是知识库中检索到的相关段落。

请你仅根据这些资料内容回答用户问题，如果没有提及，请明确说明“资料未包含该信息”。

== 知识库段落 ==
{context_text}

== 用户问题 ==
{query}

== 回答 ==
"""
    else:
        prompt = f"""
You are a professional assistant in structural engineering. You are given several reference paragraphs from a domain knowledge base.

Your job is to answer the user's question strictly based on the retrieved context below. If the answer is not present, clearly say "The knowledge base does not contain this information."

You may summarize or rephrase content, but do not invent or hallucinate.

== Retrieved Context ==
{context_text}

== User Question ==
{query}

== Your Answer ==
"""

    # GPT 生成回答
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()

    return {
        "status": "success",
        "summary": "基于知识库完成问答",
        "outputs": {
            "answer": answer
        },
        "error": None,
        "args": {
            "query": query,
            "lang": detected_lang
        }
    }

