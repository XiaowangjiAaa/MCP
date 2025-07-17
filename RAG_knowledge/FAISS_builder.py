# rag_knowledge/builder.py
import os
import json
import pickle
import numpy as np
import faiss

def build_vector_store(data_path, output_dir, lang="zh"):
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取对应语言的 embedding 和文本内容
    embedding_key = f"embedding_{lang}"
    text_key = f"text_{lang}"

    # 过滤掉缺失 embedding 的项
    valid_data = [item for item in data if embedding_key in item and item[embedding_key]]

    embeddings = np.array([item[embedding_key] for item in valid_data], dtype=np.float32)
    metadata = [item for item in valid_data]  # 保存完整的段落结构

    # 构建并保存 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(output_dir, f"crack_{lang}.index"))
    with open(os.path.join(output_dir, f"crack_{lang}_chunks.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ {lang} 向量索引与元数据已保存到 {output_dir}/crack_{lang}.*")


if __name__ == "__main__":
    DATA_PATH = r"E:\UTS_work_code\Agentic_AI\Agentic_AI_MCP_1\RAG_knowledge\regulations_embeddings\USA_crack_control_chunks_with_dual_embeddings.json"
    OUTPUT_DIR = "rag_knowledge/vector_store"

    build_vector_store(DATA_PATH, OUTPUT_DIR, lang="zh")
    build_vector_store(DATA_PATH, OUTPUT_DIR, lang="en")
