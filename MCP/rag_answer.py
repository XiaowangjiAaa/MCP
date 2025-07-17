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
        return "zh"  # default to Chinese


def format_context(contexts: List[dict], lang: str) -> str:
    text_key = "text_zh" if lang == "zh" else "text_en"
    section_key = "section_zh" if lang == "zh" else "section_en"

    lines = []
    for i, chunk in enumerate(contexts, 1):
        source = chunk.get("source", "unknown")
        section = chunk.get(section_key, "N/A")
        text = chunk.get(text_key, "").strip()
        lines.append(f"[{i}] Source: {source} | Section: {section}\n{text}")
    return "\n\n".join(lines)


@tool(name="rag_answer")
def rag_answer(query: str, lang: Optional[str] = "auto", top_k: int = 5, model: str = "gpt-4") -> dict:
    # auto-detect language
    detected_lang = detect_lang(query) if lang == "auto" else lang

    # retrieve relevant paragraphs
    contexts = retrieve_context(query, lang=detected_lang, top_k=top_k)

    # ⚠️ print retrieved content (debug)
    print("\n====== [🔍 Retrieved paragraphs] ======")
    print(format_context(contexts, detected_lang))
    print("====================================\n")

    # build prompt
    context_text = format_context(contexts, detected_lang)
    if detected_lang == "zh":
        prompt = f"""
You are a professional structural engineering assistant. Below are relevant paragraphs retrieved from the knowledge base.

Answer the user's question only using this material. If it is not mentioned, clearly state "The material does not contain this information."

== Knowledge Base Paragraphs ==
{context_text}

== User Question ==
{query}

== Answer ==
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

    # generate answer with GPT
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()

    return {
        "status": "success",
        "summary": "Answered using the knowledge base",
        "outputs": {
            "answer": answer
        },
        "error": None,
        "args": {
            "query": query,
            "lang": detected_lang
        }
    }

