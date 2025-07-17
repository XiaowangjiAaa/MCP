from RAG_knowledge.retriever import retrieve_context
from typing import List, Optional

query = "AS 3600-2018 allows designers to choose from three maximum characteristic crack widths, how many are they?"

results = retrieve_context(query, lang="en", top_k=5)

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


print("🔍 Retrieved paragraphs:", len(results))
print(format_context(results, lang="en"))
