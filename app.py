import os
import re
from typing import List, Tuple

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

import docstore

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------
# UI
# -------------------------
def apply_modern_ui():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2.4rem; max-width: 1250px; }

        section[data-testid="stSidebar"] {
            background: radial-gradient(1200px 600px at 20% 0%, rgba(124,58,237,0.22), rgba(17,26,46,0.35));
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .stButton > button {
            border-radius: 12px !important;
            padding: 0.58rem 0.95rem !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: rgba(255,255,255,0.04) !important;
            transition: all 140ms ease-in-out;
        }
        .stButton > button:hover {
            border: 1px solid rgba(124,58,237,0.65) !important;
            background: rgba(124,58,237,0.12) !important;
            transform: translateY(-1px);
        }

        input, textarea { border-radius: 12px !important; }

        details {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            background: rgba(255,255,255,0.02) !important;
            padding: 0.35rem 0.6rem;
        }

        /* Mobile-friendly tweaks (keep sidebar so Streamlit provides hamburger menu) */
        @media (max-width: 768px) {
          .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
          .stButton > button { width: 100% !important; }
          h1 { font-size: 1.65rem !important; }
          h2 { font-size: 1.35rem !important; }
          h3 { font-size: 1.1rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Secrets / API Key
# -------------------------
def get_api_key() -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or api_key
    except Exception:
        pass
    return api_key


# -------------------------
# Demo data (one-click)
# -------------------------
DEMO_DOCS = [
    (
        "AtlasDocs Demo - Support Policies.txt",
        """REFUNDS
Refunds are available within 14 days of purchase for annual plans only.
Monthly plans are not refundable.
Refund processing takes 5–7 business days.

PASSWORD RESET
To reset your password, click "Forgot Password" on the login page.
You will receive an email with a reset link.
Reset links expire after 15 minutes.

BILLING TIMING
Annual plans bill once per year on the purchase date.
Monthly plans bill on the same day each month as the original purchase date.
If a billing attempt fails, we retry automatically within 24 hours.
""",
    ),
    (
        "AtlasDocs Demo - Employee Onboarding.txt",
        """ONBOARDING OVERVIEW
New hires should complete onboarding within the first 5 business days.

ACCOUNTS AND ACCESS
Request access to tools through the IT portal.
Multi-factor authentication is required for email and the password manager.

EQUIPMENT
Company laptops are issued within 2 business days of start date.
If equipment is delayed, contact IT support.

TIME OFF
Time off requests should be submitted at least 2 weeks in advance when possible.
""",
    ),
    (
        "AtlasDocs Demo - Troubleshooting Guide.txt",
        """APP CRASHING
If the app is crashing, update to the latest version.
If issues continue, reinstall the app and restart your device.

LOGIN ISSUES
If you cannot log in, confirm your email address is correct.
Use "Forgot Password" to reset your password.
If you do not receive the reset email within 10 minutes, check spam or contact support.

PERFORMANCE
Close background apps to improve performance.
If you are on a slow network, try switching to a stable Wi-Fi connection.
""",
    ),
]


def load_demo_docs_into_library() -> int:
    """Adds demo docs into the saved library (idempotent). Returns count of newly-added docs."""
    existing_ids = {d["doc_id"] for d in docstore.list_docs()}
    added = 0

    for filename, text in DEMO_DOCS:
        chunks = chunk_text(text)
        if not chunks:
            continue
        embs = embed_chunks(chunks)

        doc_id = docstore.sha16(filename + "::" + text[:20000])
        if doc_id in existing_ids:
            continue

        docstore.upsert_doc(doc_id, filename)
        docstore.save_raw(doc_id, text)
        docstore.save_index(doc_id, chunks, embs)
        added += 1

    return added


# -------------------------
# Document parsing
# -------------------------
def read_uploaded_file(file) -> str:
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    return ""


def chunk_text(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Split on ALL CAPS headings
    sections = re.split(r"\n(?=[A-Z][A-Z ]+\n)", text)
    chunks = [s.strip() for s in sections if s.strip()]

    # Fallback chunking if no headings
    if len(chunks) <= 1:
        words = text.split()
        out, buf = [], []
        for w in words:
            buf.append(w)
            if len(buf) >= 220:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
        chunks = out

    return chunks


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_chunks(chunks: List[str]) -> np.ndarray:
    model = load_embedder()
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def build_knn(embeddings: np.ndarray) -> NearestNeighbors:
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)
    return nn


def retrieve_across_docs(
    query: str,
    doc_ids: List[str],
    top_k: int,
) -> List[Tuple[float, str, int, str, str]]:
    """
    Returns list of (similarity, doc_id, chunk_index, doc_name, chunk_text)
    """
    model = load_embedder()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    all_chunks: List[str] = []
    all_meta: List[Tuple[str, int, str]] = []  # (doc_id, chunk_idx, doc_name)
    all_embs: List[np.ndarray] = []

    docs = {d["doc_id"]: d["name"] for d in docstore.list_docs()}

    for doc_id in doc_ids:
        loaded = docstore.load_index(doc_id)
        if not loaded:
            continue
        chunks, embs = loaded
        all_embs.append(embs)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_meta.append((doc_id, i, docs.get(doc_id, doc_id)))

    if not all_chunks:
        return []

    E = np.vstack(all_embs)
    nn = build_knn(E)

    k = min(top_k, len(all_chunks))
    distances, idxs = nn.kneighbors(q_emb, n_neighbors=k)

    hits = []
    for dist, idx in zip(distances[0], idxs[0]):
        sim = 1.0 - float(dist)
        doc_id, chunk_idx, doc_name = all_meta[int(idx)]
        hits.append((sim, doc_id, chunk_idx, doc_name, all_chunks[int(idx)]))
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits


# -------------------------
# Smart suggestions
# -------------------------
@st.cache_data(show_spinner=False)
def generate_suggested_questions(doc_key: str, doc_preview: str, model_name: str) -> List[str]:
    client = OpenAI()
    system = "Return ONLY 3 to 6 helpful user questions as a bullet list. No extra text."
    user = f"""DOCUMENT PREVIEW:
{doc_preview}

Write 3-6 realistic questions someone would ask about this document.
"""
    resp = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    text = resp.output_text.strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    qs = []
    for l in lines:
        l = re.sub(r"^[\\-\\*\\d\\.\\)\\s]+", "", l).strip()
        if not l:
            continue
        if not l.endswith("?"):
            l += "?"
        qs.append(l)

    out, seen = [], set()
    for q in qs:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)

    return out[:6] if out else [
        "What is this document about?",
        "What are the key policies or rules?",
        "Are there deadlines or time limits mentioned?",
    ]


def doc_preview_from_chunks(chunks: List[str], max_chars: int = 4500) -> str:
    ranked = sorted(chunks, key=len, reverse=True)[:6]
    preview = "\n\n---\n\n".join(ranked)
    return preview[:max_chars]


# -------------------------
# RAG synthesis
# -------------------------
def synthesize_answer(question: str, sources: List[Tuple[int, str, str]], model_name: str) -> str:
    """
    sources = [(num, doc_name, text)]
    """
    client = OpenAI()

    ctx = []
    for num, doc_name, text in sources:
        ctx.append(f"[{num}] ({doc_name})\n{text}")
    context = "\n\n---\n\n".join(ctx)

    system = (
        "You are a helpful assistant. Answer using ONLY the DOCUMENT SOURCES.\n"
        "Cite facts like [1], [2]. If not found, say: "
        "\"I don't know based on the provided documents.\""
    )

    user = f"""DOCUMENT SOURCES:
{context}

QUESTION:
{question}

Return in this format:
TL;DR: <one sentence>
Answer: <bullets>
Details: <short paragraph if needed>
"""
    resp = client.responses.create(
        model=model_name,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.output_text.strip()


def compute_confidence(similarities: List[float], answer: str) -> str:
    if not similarities:
        return "Low"
    top = similarities[0]
    avg = sum(similarities) / len(similarities)
    a = answer.lower()
    if "i don't know" in a:
        return "Low"
    if top >= 0.55 and avg >= 0.35:
        return "High"
    if top >= 0.40:
        return "Medium"
    return "Low"


# -------------------------
# Pages
# -------------------------
def landing_page():
    st.markdown(
        """
        <style>
        .hero {
            padding: 26px 24px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.10);
            background:
              radial-gradient(800px 400px at 10% 0%, rgba(124,58,237,0.30), rgba(0,0,0,0)),
              linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        }
        .badge {
            display:inline-flex; align-items:center; gap:8px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.05);
            font-size: 12px;
            opacity: 0.92;
        }
        .grid3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 16px;
        }
        .card {
            padding: 16px 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.09);
            background: rgba(255,255,255,0.03);
        }
        .muted { opacity: 0.80; }
        .preview {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            padding: 16px;
        }
        .kbd {
            display:inline-block;
            padding: 2px 8px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
            font-size: 12px;
        }
        @media (max-width: 900px) {
            .grid3 { grid-template-columns: 1fr; }
        }
        </style>

        <div class="hero">
          <div class="badge">⚡ True RAG • Multi-doc • PDF • Analytics</div>
          <h1 style="margin-top:14px; margin-bottom:6px;">AtlasDocs</h1>
          <h3 style="margin-top:0px; font-weight:600;">
            Your docs, instantly searchable — answers with citations, not guesses.
          </h3>
          <div class="muted" style="margin-top:10px;">
            Upload policies, SOPs, FAQs, onboarding docs, contracts — then chat with a grounded assistant
            that shows exactly where every answer came from.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### What you can do")
        st.markdown(
            """
- **Upload PDFs/TXTs** to a saved library  
- **Search across multiple documents** at once  
- **Cited answers** with source snippets  
- **Smart suggested questions** based on your document  
- **Analytics dashboard** (questions per day, recent questions)  
            """
        )

        st.markdown("### Quick start")
        st.markdown(
            """
1) Go to **Documents** → click **✨ Load demo docs** (or upload a PDF/TXT)  
2) Go to **Demo** → select docs → ask a question  
3) Check **Analytics** after a few questions  
            """
        )

    with col2:
        st.markdown("### Product preview")
        st.markdown(
            """
            <div class="preview">
              <div class="muted" style="margin-bottom:10px;">Mobile-friendly: Chat / Sources tabs</div>
              <div class="muted">Suggested questions adapt to the selected document.</div>
              <div style="margin-top:12px;">
                <span class="kbd">What is the refund policy?</span>
                <span class="kbd">How do I reset my password?</span>
                <span class="kbd">When is billing?</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Why this matters")
    st.markdown(
        """
        <div class="grid3">
          <div class="card"><b>Grounded answers</b><br/><span class="muted">Only uses retrieved text from your docs.</span></div>
          <div class="card"><b>Multi-doc search</b><br/><span class="muted">Ask questions across several documents at once.</span></div>
          <div class="card"><b>Mobile friendly</b><br/><span class="muted">Tabs prevent cramped layouts on phones.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def docs_page():
    st.header("📚 Documents (Saved Library)")

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("✨ Load demo docs", use_container_width=True):
            with st.spinner("Loading demo documents..."):
                added = load_demo_docs_into_library()
            if added == 0:
                st.info("Demo docs are already loaded.")
            else:
                st.success(f"Loaded {added} demo document(s).")
            st.rerun()
    with colB:
        st.caption("Instant content to test the app without uploading anything.")

    upload = st.file_uploader("Upload a document", type=["txt", "pdf"])
    if upload:
        raw = read_uploaded_file(upload)
        if not raw.strip():
            st.error("Could not read document (empty or unreadable).")
            return

        chunks = chunk_text(raw)
        embs = embed_chunks(chunks)

        doc_id = docstore.sha16(upload.name + "::" + raw[:20000])
        docstore.upsert_doc(doc_id, upload.name)
        docstore.save_raw(doc_id, raw)
        docstore.save_index(doc_id, chunks, embs)

        st.success(f"Saved: {upload.name} (chunks: {len(chunks)})")

    docs = docstore.list_docs()
    if not docs:
        st.info("No saved docs yet. Upload a PDF/TXT above, or click **✨ Load demo docs**.")
        return

    st.subheader("Saved docs")
    for d in docs:
        c1, c2, c3 = st.columns([4, 2, 1])
        with c1:
            st.write(f"**{d['name']}**")
            st.caption(f"doc_id: {d['doc_id']}")
        with c2:
            st.caption(f"Added: {d['created_at']}")
        with c3:
            if st.button("Delete", key=f"del_{d['doc_id']}"):
                docstore.delete_doc(d["doc_id"])
                st.rerun()


def demo_page(user_label: str):
    st.title("🧠 AtlasDocs Demo")
    st.caption("Mobile-friendly: Chat and Sources are in tabs. Sidebar becomes a hamburger on mobile.")

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not set in Streamlit Secrets.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    docs = docstore.list_docs()
    if not docs:
        st.info("No saved documents yet. Go to **Documents** and upload a PDF/TXT (or load demo docs).")
        return

    doc_map = {d["name"]: d["doc_id"] for d in docs}
    names = list(doc_map.keys())

    # Keep sidebar controls (hamburger on mobile)
    st.sidebar.header("Search Scope")
    selected_names = st.sidebar.multiselect("Search these documents", options=names, default=names[:1])
    selected_doc_ids = [doc_map[n] for n in selected_names] if selected_names else []

    st.sidebar.header("Model")
    model_name = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
    top_k = st.sidebar.slider("Sources to retrieve", 1, 8, 4)

    st.sidebar.header("Analytics label")
    st.sidebar.caption("Optional: used only for analytics grouping.")
    user_label = st.sidebar.text_input("Name", value=user_label)

    # State
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "last_hits" not in st.session_state:
        st.session_state.last_hits = []
    if "last_conf" not in st.session_state:
        st.session_state.last_conf = None

    base_doc = selected_doc_ids[0] if selected_doc_ids else docs[0]["doc_id"]
    loaded = docstore.load_index(base_doc)
    if loaded:
        base_chunks, _ = loaded
        preview = doc_preview_from_chunks(base_chunks)
        suggested = generate_suggested_questions(doc_key=base_doc, doc_preview=preview, model_name=model_name)
    else:
        suggested = ["What is this document about?", "What are key policies?", "Any deadlines?"]

    def ask(q: str):
        st.session_state.pending = (q or "").strip()

    # Suggested prompts: 1-column (better on mobile)
    st.markdown("### Suggested questions")
    for i, q in enumerate(suggested):
        if st.button(q, use_container_width=True, key=f"sugg_{i}"):
            ask(q)

    tab_chat, tab_sources = st.tabs(["💬 Chat", "📚 Sources"])

    with tab_chat:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        question = st.text_input("Ask a question", placeholder="Type your question…")
        if st.button("Send", use_container_width=True):
            ask(question)

        pending = st.session_state.pending
        if pending:
            q = pending
            st.session_state.pending = None

            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.write(q)

            if not selected_doc_ids:
                st.error("Select at least one document in the sidebar (hamburger on mobile).")
                return

            docstore.log_event(user=user_label or "anonymous", doc_ids=selected_doc_ids, question=q)

            hits = retrieve_across_docs(q, selected_doc_ids, top_k=top_k)
            st.session_state.last_hits = hits

            sources = []
            sims = []
            for i, (sim, _doc_id, _chunk_idx, doc_name, chunk_text) in enumerate(hits, start=1):
                sources.append((i, doc_name, chunk_text))
                sims.append(sim)

            with st.spinner("Thinking..."):
                answer = synthesize_answer(q, sources, model_name=model_name)

            conf = compute_confidence(sims, answer)
            st.session_state.last_conf = conf

            st.session_state.chat.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
                if conf == "High":
                    st.success("Confidence: HIGH")
                elif conf == "Medium":
                    st.warning("Confidence: MEDIUM")
                else:
                    st.error("Confidence: LOW")

    with tab_sources:
        st.subheader("Sources")
        if not st.session_state.last_hits:
            st.info("Ask a question to see sources.")
        else:
            st.caption(f"Confidence: **{st.session_state.last_conf}**")
            for i, (sim, doc_id, chunk_idx, doc_name, chunk_text) in enumerate(st.session_state.last_hits, start=1):
                with st.expander(f"[{i}] {doc_name} • similarity {sim:.3f}", expanded=(i == 1)):
                    st.write(chunk_text)


def analytics_page():
    st.header("📈 Analytics")

    docstore.init_db()
    import sqlite3

    with sqlite3.connect(docstore.DB_PATH) as conn:
        df = pd.read_sql_query("SELECT ts, user, doc_ids, question FROM events ORDER BY ts ASC", conn)

    if df.empty:
        st.info("No analytics yet. Ask some questions in Demo.")
        return

    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date

    st.metric("Total questions", int(len(df)))

    st.subheader("Questions per day")
    per_day = df.groupby("date").size().reset_index(name="questions")
    fig = plt.figure()
    plt.plot(per_day["date"], per_day["questions"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top user labels")
    top_users = df.groupby("user").size().sort_values(ascending=False).head(20)
    st.dataframe(top_users.reset_index(name="questions"), use_container_width=True)

    st.subheader("Recent questions")
    st.dataframe(df.tail(50), use_container_width=True)


# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title="AtlasDocs", page_icon="🧠", layout="wide")
    apply_modern_ui()
    docstore.init_db()

    st.sidebar.title("AtlasDocs")
    page = st.sidebar.radio("Navigate", ["Landing", "Documents", "Demo", "Analytics"], index=2)

    if "user_label" not in st.session_state:
        st.session_state["user_label"] = "anonymous"

    if page == "Landing":
        landing_page()
    elif page == "Documents":
        docs_page()
    elif page == "Demo":
        demo_page(user_label=st.session_state["user_label"])
    else:
        analytics_page()


if __name__ == "__main__":
    main()
