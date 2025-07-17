# rag_knowledge/retriever.py
from dotenv import load_dotenv
import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

# initialize OpenAI client (requires API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# vector store directory
VECTOR_STORE_DIR = "rag_knowledge/vector_store"

# available country/language combinations
AVAILABLE_COUNTRIES = ["USA", "AU"]
AVAILABLE_LANGS = ["zh", "en"]


def load_vector_store(country: str = "USA", lang: str = "zh"):
    """Load vector index and paragraphs for the given country and language.
    File format: {country}_crack_{lang}.index / .pkl"""
    # sanity checks
    if country not in AVAILABLE_COUNTRIES:
        raise ValueError(f"Unsupported country: {country}. Available: {AVAILABLE_COUNTRIES}")
    if lang not in AVAILABLE_LANGS:
        raise ValueError(f"Unsupported language: {lang}. Available: {AVAILABLE_LANGS}")

    index_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}_chunks.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Missing vector file or paragraph data:\n{index_path}\n{chunks_path}")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def embed_query(text: str) -> np.ndarray:
    """Call OpenAI embedding API to convert text into a vector."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    return embedding.reshape(1, -1)


def retrieve_context(query: str, country: str = "USA", lang: str = "zh", top_k: int = 5):
    """Retrieve top-k relevant paragraphs from the knowledge base."""
    index, chunks = load_vector_store(country, lang)
    query_vec = embed_query(query)
    D, I = index.search(query_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results


def pretty_print(results, lang="zh"):
    """Pretty-print results for debugging/testing."""
    text_key = "text_zh" if lang == "zh" else "text_en"
    section_key = "section_zh" if lang == "zh" else "section_en"

    for i, chunk in enumerate(results, 1):
        print(f"\n[Chunk {i}] Source: {chunk.get('source', 'unknown')} | Section: {chunk.get(section_key, 'N/A')}")
        print(chunk.get(text_key, "")[:500])  # preview length adjustable


def list_available_knowledge_bases():
    """List available country/language combinations in the vector store."""
    combos = []
    for country in AVAILABLE_COUNTRIES:
        for lang in AVAILABLE_LANGS:
            index_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}.index")
            chunks_path = os.path.join(VECTOR_STORE_DIR, f"{country}_crack_{lang}_chunks.pkl")
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                combos.append((country, lang))
    return combos
