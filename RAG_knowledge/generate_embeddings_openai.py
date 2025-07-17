from openai import OpenAI
import json
import time
import os
from dotenv import load_dotenv

# initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# file paths
INPUT_PATH = r"E:\UTS_work_code\Agentic_AI\Agentic_AI_MCP_1\RAG_knowledge\regulations_data\USA_crack_control_chunks_translated_en.json"
OUTPUT_PATH = r"E:\UTS_work_code\Agentic_AI\Agentic_AI_MCP_1\RAG_knowledge\regulations_embeddings\USA_crack_control_chunks_with_dual_embeddings.json"
ERROR_LOG_PATH = r"E:\UTS_work_code\Agentic_AI\Agentic_AI_MCP\RAG\embedding_dual_errors.json"

# load bilingual structured text
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedded_chunks = []
error_chunks = []

for idx, chunk in enumerate(chunks):
    try:
        # Chinese embedding
        embedding_zh = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk["text_zh"]
        ).data[0].embedding
        chunk["embedding_zh"] = embedding_zh

        # English embedding
        embedding_en = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk["text_en"]
        ).data[0].embedding
        chunk["embedding_en"] = embedding_en

        embedded_chunks.append(chunk)
        print(f"[{idx+1}/{len(chunks)}] ✅ Embedded (ZH + EN)")
        time.sleep(1)

    except Exception as e:
        print(f"[{idx+1}] ❌ Error embedding chunk: {e}")
        chunk["error"] = str(e)
        error_chunks.append(chunk)
        continue

# save embeddings with both languages
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

# save error log
if error_chunks:
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(error_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n⚠️ {len(error_chunks)} chunks failed. See {ERROR_LOG_PATH}")

print(f"\n🎉 All done! {len(embedded_chunks)} chunks with dual embeddings saved to:\n{OUTPUT_PATH}")
