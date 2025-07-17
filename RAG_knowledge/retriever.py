# rag_knowledge/retriever.py
from dotenv import load_dotenv
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

# 初始化 OpenAI 客户端（需确保已设置 API 密钥）
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 向量库目录
VECTOR_STORE_DIR = "rag_knowledge/vector_store"

# 可用国家/语言组合（可按需扩展）
AVAILABLE_COUNTRIES = ["USA", "AU"]
AVAILABLE_LANGS = ["zh", "en"]


def load_vector_store(country: str = "USA", lang: str = "zh"):
    """
    根据国家和语言加载向量索引与文本段落。
    支持文件名格式: {country}_crack_{lang}.index / .pkl
    """
    # 合法性检查
    if country not in AVAILABLE_COUNTRIES:
        raise ValueError(f"❌ 不支持的国家: {country}，当前支持: {AVAILABLE_COUNTRIES}")
    if lang not in AVAILABLE_LANGS:
        raise ValueError(f"❌ 不支持的语言: {lang}，当前支持: {AVAILABLE_LANGS}")

    index_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}_chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"❌ 缺失向量文件或段落数据: \n{index_path}\n{chunks_path}")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def embed_query(text: str) -> np.ndarray:
    """
    调用 OpenAI embedding API 将文本转为向量
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    return embedding.reshape(1, -1)


def retrieve_context(query: str, country: str = "USA", lang: str = "zh", top_k: int = 5):
    """
    主函数：根据用户查询，从指定国家/语言的知识库中检索 top-k 相关段落
    """
    index, chunks = load_vector_store(country, lang)
    query_vec = embed_query(query)
    D, I = index.search(query_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results


def pretty_print(results, lang="zh"):
    """
    调试/测试用：以人类可读格式输出结果段落
    """
    text_key = "text_zh" if lang == "zh" else "text_en"
    section_key = "section_zh" if lang == "zh" else "section_en"

    for i, chunk in enumerate(results, 1):
        print(f"\n【段落 {i}】来源: {chunk.get('source', '未知')} | 章节: {chunk.get(section_key, 'N/A')}")
        print(chunk.get(text_key, "")[:500])  # 可调整字符预览长度


def list_available_knowledge_bases():
    """
    自动列出当前向量库中可用的国家+语言组合
    """
    combos = []
    for country in AVAILABLE_COUNTRIES:
        for lang in AVAILABLE_LANGS:
            index_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}.index")
            chunks_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}_chunks.pkl")
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                combos.append((country, lang))
    return combos
