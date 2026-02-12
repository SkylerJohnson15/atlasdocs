import os
import re
import json
import pickle
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Try FAISS first (fast). If not installed, we fall back to scikit-learn.
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors


DATA_PATH = os.path.join("data", "company_docs.txt")
STORAGE_DIR = "storage"

CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(STORAGE_DIR, "embeddings.npy")

INDEX_META_PATH = os.path.join(STORAGE_DIR, "index_meta.json")
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
SKLEARN_INDEX_PATH = os.path.join(STORAGE_DIR, "sklearn_index.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            f"Expected at: {os.path.abspath(path)}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> List[str]:
    """
    Section-based chunking:
    Splits when a line looks like an ALL CAPS header (e.g., REFUNDS, BILLING).
    Produces clean chunks so answers aren't cut off.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    # Split before ALL CAPS headers followed by newline
    sections = re.split(r"\n(?=[A-Z][A-Z ]+\n)", text)

    chunks: List[str] = []
    for sec in sections:
        cleaned = sec.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def ensure_storage_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # type: ignore
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)  # type: ignore


def build_sklearn_index(embeddings: np.ndarray):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)
    with open(SKLEARN_INDEX_PATH, "wb") as f:
        pickle.dump(nn, f)


def main():
    ensure_storage_dir()

    print("Loading docs...")
    text = read_text_file(DATA_PATH)
    print(f"Doc length (characters): {len(text)}")

    print("Chunking docs...")
    chunks = split_into_chunks(text)
    print(f"Chunks created: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError(
            "No chunks were created. data/company_docs.txt is empty (or only whitespace)."
        )

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = embed_chunks(chunks, model)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([{"id": i, "text": c} for i, c in enumerate(chunks)], f, ensure_ascii=False, indent=2)

    np.save(EMBEDDINGS_PATH, embeddings)

    if HAS_FAISS:
        print("Building FAISS index...")
        build_faiss_index(embeddings)
        index_type = "faiss"
    else:
        print("FAISS not available -> Building scikit-learn index...")
        build_sklearn_index(embeddings)
        index_type = "sklearn"

    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_type": index_type,
                "embed_model_name": MODEL_NAME,
                "chunking": "section_headers",
            },
            f,
            indent=2,
        )

    print("✅ Ingestion complete!")
    print(f"Index type: {index_type}")
    print("Files saved in /storage")


if __name__ == "__main__":
    main()
