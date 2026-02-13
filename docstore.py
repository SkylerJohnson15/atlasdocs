import os
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
DB_PATH = os.path.join(DATA_DIR, "atlasdocs.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def sha16(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user TEXT NOT NULL,
                doc_ids TEXT NOT NULL,
                question TEXT NOT NULL
            )
            """
        )
        conn.commit()


def upsert_doc(doc_id: str, name: str):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO docs (doc_id, name, created_at) VALUES (?, ?, ?)",
            (doc_id, name, datetime.utcnow().isoformat()),
        )
        conn.commit()


def list_docs() -> List[Dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT doc_id, name, created_at FROM docs ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [{"doc_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def delete_doc(doc_id: str):
    init_db()
    # delete DB row
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        conn.commit()

    # delete files
    raw_path = os.path.join(UPLOADS_DIR, f"{doc_id}.txt")
    chunks_path = os.path.join(INDEX_DIR, f"{doc_id}_chunks.json")
    emb_path = os.path.join(INDEX_DIR, f"{doc_id}_embeddings.npy")
    for p in [raw_path, chunks_path, emb_path]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def save_raw(doc_id: str, text: str):
    path = os.path.join(UPLOADS_DIR, f"{doc_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_raw(doc_id: str) -> Optional[str]:
    path = os.path.join(UPLOADS_DIR, f"{doc_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_index(doc_id: str, chunks: List[str], embeddings: np.ndarray):
    chunks_path = os.path.join(INDEX_DIR, f"{doc_id}_chunks.json")
    emb_path = os.path.join(INDEX_DIR, f"{doc_id}_embeddings.npy")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(emb_path, embeddings.astype(np.float32))


def load_index(doc_id: str) -> Optional[Tuple[List[str], np.ndarray]]:
    chunks_path = os.path.join(INDEX_DIR, f"{doc_id}_chunks.json")
    emb_path = os.path.join(INDEX_DIR, f"{doc_id}_embeddings.npy")
    if not os.path.exists(chunks_path) or not os.path.exists(emb_path):
        return None

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(emb_path).astype(np.float32)
    return chunks, embeddings


def log_event(user: str, doc_ids: List[str], question: str):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO events (ts, user, doc_ids, question) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), user, ",".join(doc_ids), question),
        )
        conn.commit()


def analytics_summary() -> Dict:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM events")
        total_q = cur.fetchone()[0]

        cur.execute("SELECT user, COUNT(*) c FROM events GROUP BY user ORDER BY c DESC LIMIT 20")
        by_user = cur.fetchall()

        cur.execute("SELECT doc_ids, COUNT(*) c FROM events GROUP BY doc_ids ORDER BY c DESC LIMIT 20")
        by_docset = cur.fetchall()

    return {
        "total_questions": total_q,
        "by_user": by_user,
        "by_docset": by_docset,
    }
