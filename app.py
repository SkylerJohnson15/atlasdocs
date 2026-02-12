import os
import re
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Vector search: prefer FAISS, fallback to scikit-learn
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors


# -------------------------
# Core Settings
# -------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------
# UI Polish
# -------------------------
def apply_modern_ui():
    st.markdown(
        """
        <style>
        /* Layout width + spacing */
        .block-container { padding-top: 1.25rem; padding-bottom: 2.4rem; max-width: 1250px; }

        /* Sidebar gradient + border */
        section[data-testid="stSidebar"] {
            background: radial-gradient(1200px 600px at 20% 0%, rgba(124,58,237,0.22), rgba(17,26,46,0.35));
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Buttons */
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

        /* Inputs */
        input, textarea {
            border-radius: 12px !important;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            background: rgba(255,255,255,0.03) !important;
        }

        /* Expanders */
        details {
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            background: rgba(255,255,255,0.02) !important;
            padding: 0.35rem 0.6rem;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.06);
            background: rgba(255,255,255,0.02);
            padding: 0.2rem 0.25rem;
        }

        /* Landing components */
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
        """,
        unsafe_allow_html=True
    )


# -------------------------
# Embeddings / Index
# -------------------------
@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def chunk_text(text: str) -> List[str]:
    """Split by ALL CAPS headers for decent chunking on policy/FAQ docs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    sections = re.split(r"\n(?=[A-Z][A-Z ]+\n)", text)
    return [s.strip() for s in sections if s.strip()]


def build_index(chunks: List[str]):
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    if HAS_FAISS:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # type: ignore
        index.add(embeddings)
        return index, "faiss"
    else:
        nn = NearestNeighbors(n_neighbors=min(5, len(chunks)), metric="cosine")
        nn.fit(embeddings)
        return nn, "sklearn"


def retrieve(query: str, chunks: List[str], index, index_type: str, top_k: int) -> List[Tuple[float, int, str]]:
    """Return list of (similarity, chunk_id, chunk_text) sorted best->worst."""
    top_k = min(top_k, len(chunks))
    if top_k <= 0:
        return []

    model = load_embedding_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    results: List[Tuple[float, int, str]] = []

    if index_type == "faiss":
        scores, idxs = index.search(q_emb, top_k)  # type: ignore
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((float(score), int(idx), chunks[int(idx)]))
    else:
        distances, idxs = index.kneighbors(q_emb, n_neighbors=top_k)
        for dist, idx in zip(distances[0], idxs[0]):
            sim = 1.0 - float(dist)
            results.append((sim, int(idx), chunks[int(idx)]))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


def compute_confidence(similarities: List[float], answer: str) -> str:
    """Simple heuristic confidence."""
    if not similarities:
        return "Low"

    ans = (answer or "").lower()
    if "i don't know" in ans or "i do not know" in ans:
        return "Low"

    top = similarities[0]
    avg = sum(similarities) / len(similarities)

    if top >= 0.55 and avg >= 0.35:
        return "High"
    if top >= 0.40:
        return "Medium"
    return "Low"


def followup_suggestions(question: str) -> List[str]:
    q = question.strip().rstrip("?")
    return [
        f"Can you point me to the exact section about {q}?",
        "Is there an exception or edge case for this policy?",
        "What steps should I follow next?",
    ]


def synthesize_answer(
    question: str,
    sources: List[Tuple[int, str]],
    chat_history: List[Dict],
    model_name: str
) -> str:
    """LLM synthesis (True RAG): answer only from sources + cite like [1], [2]."""
    client = OpenAI()

    trimmed_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
    history_text = ""
    for msg in trimmed_history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    context_blocks = []
    for num, text in sources:
        context_blocks.append(f"[{num}]\n{text}")
    context = "\n\n---\n\n".join(context_blocks)

    system_msg = (
        "You are a helpful company support assistant. "
        "Answer using ONLY the provided DOCUMENT SOURCES. "
        "When you use a fact, cite it like [1] or [2]. "
        "Return the answer in this format:\n"
        "TL;DR: <one sentence>\n"
        "Answer: <2-6 bullet points if possible>\n"
        "Details: <short paragraph if needed>\n"
        "If the answer is not in sources, say exactly: "
        "\"I don't know based on the provided documents.\""
    )

    user_msg = f"""DOCUMENT SOURCES:
{context}

CHAT HISTORY:
{history_text}

CURRENT QUESTION:
{question}

RULES:
- Use ONLY DOCUMENT SOURCES.
- Cite facts like [1], [2].
- If missing, say: "I don't know based on the provided documents."
- Do not invent policies, steps, or facts.
"""

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.output_text.strip()


def export_chat_txt(chat_history: List[Dict]) -> str:
    lines = []
    for msg in chat_history:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines).strip()


# -------------------------
# Landing Page
# -------------------------
def landing_page():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">⚡ True RAG • Citations • Confidence</div>
          <h1 style="margin-top:14px; margin-bottom:6px;">AtlasDocs</h1>
          <h3 style="margin-top:0px; font-weight:600;">Your company’s docs, instantly searchable — answers with citations, not guesses.</h3>
          <div class="muted" style="margin-top:10px;">
            Upload internal policies, FAQs, onboarding docs, and get a grounded assistant that shows where every answer came from.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    col1, col2 = st.columns([1.15, 1], gap="large")
    with col1:
        st.markdown("### What you get")
        st.markdown(
            """
- **Upload** a document (policies / SOPs / FAQs)
- **Chat** like a support agent
- **Citations** to the exact source chunks
- **Confidence score** to reduce hallucinations
- **Source panel** like Perplexity (modern UX)
            """
        )
        st.markdown(
            """
<div class="muted">
Tip: In the demo, try <span class="kbd">Refund policy</span> or <span class="kbd">Password reset</span>.
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### How it works")
        st.markdown(
            """
1) Chunk the document into sections  
2) Embed sections into vectors  
3) Retrieve top-k relevant chunks (semantic search)  
4) LLM synthesizes an answer grounded in sources  
            """
        )

    with col2:
        st.markdown("### Product preview")
        st.markdown(
            """
            <div class="preview">
              <div class="muted" style="margin-bottom:10px;">Chat + Sources layout</div>
              <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:12px; background: rgba(255,255,255,0.02);">
                <div class="muted" style="font-size:12px;">User</div>
                <div style="margin-top:6px;">What is the refund policy?</div>
              </div>

              <div style="height:10px;"></div>

              <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:12px; background: rgba(255,255,255,0.02);">
                <div class="muted" style="font-size:12px;">AtlasDocs</div>
                <div style="margin-top:6px;">
                  TL;DR: Annual plans are refundable within 14 days. Monthly plans are not refundable. [1]
                </div>
              </div>

              <div style="margin-top:12px;" class="muted">
                Sources panel shows similarity scores and supporting text.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Why teams like it")
    st.markdown(
        """
        <div class="grid3">
          <div class="card"><b>Grounded answers</b><br/><span class="muted">Uses only retrieved sources.</span></div>
          <div class="card"><b>Citations</b><br/><span class="muted">Every answer references the doc chunks.</span></div>
          <div class="card"><b>Modern UX</b><br/><span class="muted">Chat + Sources panel looks like a real SaaS.</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    st.info("Open **Demo** in the sidebar to try it.")


# -------------------------
# Demo Page (2-panel)
# -------------------------
def demo_page():
    st.title("🧠 AtlasDocs Demo")
    st.caption("Modern RAG assistant: chat on the left, sources on the right.")

    # ✅ Streamlit Cloud + Local compatible key loading
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY is not set.\n\n"
            "Local (Windows): set OPENAI_API_KEY in your terminal.\n"
            "Streamlit Cloud: add OPENAI_API_KEY in app Secrets."
        )
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_doc_name" not in st.session_state:
        st.session_state.active_doc_name = None
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None
    if "last_hits" not in st.session_state:
        st.session_state["last_hits"] = []
    if "last_confidence" not in st.session_state:
        st.session_state["last_confidence"] = None
    if "last_question" not in st.session_state:
        st.session_state["last_question"] = None

    st.sidebar.header("Demo Settings")
    openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
    top_k_ui = st.sidebar.slider("Sources to retrieve", 1, 5, 3)
    show_sources = st.sidebar.checkbox("Show sources panel", True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state["question_input"] = ""
        st.session_state["pending_question"] = None
        st.session_state["last_hits"] = []
        st.session_state["last_confidence"] = None
        st.session_state["last_question"] = None

    txt = export_chat_txt(st.session_state.chat_history)
    filename = f"atlasdocs_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.sidebar.download_button("Export chat (.txt)", data=txt, file_name=filename, mime="text/plain")

    uploaded_file = st.file_uploader("Upload a TXT document", type=["txt"])
    if not uploaded_file:
        st.info("Upload a TXT file to start.")
        return

    # New document uploaded -> reset conversation
    if st.session_state.active_doc_name != uploaded_file.name:
        st.session_state.active_doc_name = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state["question_input"] = ""
        st.session_state["pending_question"] = None
        st.session_state["last_hits"] = []
        st.session_state["last_confidence"] = None
        st.session_state["last_question"] = None

    text = uploaded_file.read().decode("utf-8", errors="replace")
    chunks = chunk_text(text)
    if not chunks:
        st.error("Document appears empty or couldn't be parsed.")
        return

    index, index_type = build_index(chunks)
    top_k = min(top_k_ui, len(chunks))
    st.success(f"Indexed {len(chunks)} chunks. Retrieving top {top_k} sources.")

    def submit_question(text: str):
        st.session_state["pending_question"] = (text or "").strip()
        st.session_state["question_input"] = ""

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("### Try these example prompts")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Refund policy", use_container_width=True):
                submit_question("What is the refund policy?")
        with c2:
            if st.button("Password reset", use_container_width=True):
                submit_question("How do I reset my password?")
        with c3:
            if st.button("Billing timing", use_container_width=True):
                submit_question("When does billing occur?")

        st.markdown("### Chat")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        st.markdown("### Ask a question")

        def on_send():
            submit_question(st.session_state.get("question_input", ""))

        st.text_input("Question", key="question_input")
        st.button("Send", on_click=on_send)

        pending = st.session_state.get("pending_question")
        if pending:
            question = pending
            st.session_state["pending_question"] = None
            st.session_state["last_question"] = question

            # Render user bubble immediately
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            hits = retrieve(question, chunks, index, index_type, top_k)
            similarities = [h[0] for h in hits]
            numbered_sources = [(i, chunk) for i, (_sim, _id, chunk) in enumerate(hits, start=1)]

            with st.spinner("Thinking..."):
                answer = synthesize_answer(
                    question=question,
                    sources=numbered_sources,
                    chat_history=st.session_state.chat_history,
                    model_name=openai_model,
                )

            confidence = compute_confidence(similarities, answer)

            # Save for sources panel
            st.session_state["last_hits"] = hits
            st.session_state["last_confidence"] = confidence

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.write(answer)

                if confidence == "High":
                    st.success("Confidence: HIGH")
                elif confidence == "Medium":
                    st.warning("Confidence: MEDIUM")
                else:
                    st.error("Confidence: LOW")
                    st.markdown("**Try a follow-up:**")
                    for sug in followup_suggestions(question):
                        st.markdown(f"- {sug}")

    with right:
        st.markdown("### Sources")
        if not show_sources:
            st.info("Sources panel is hidden (enable it in the sidebar).")
        else:
            last_q = st.session_state.get("last_question")
            last_hits = st.session_state.get("last_hits", [])
            last_conf = st.session_state.get("last_confidence")

            if last_q:
                st.caption(f"Showing sources for: **{last_q}**")
            if last_conf:
                st.caption(f"Confidence: **{last_conf}**")

            if not last_hits:
                st.info("Ask a question to see relevant sources here.")
            else:
                for i, (score, _chunk_id, chunk_text_val) in enumerate(last_hits, start=1):
                    with st.expander(f"Source {i} • similarity {score:.3f}", expanded=(i == 1)):
                        st.write(chunk_text_val)


def main():
    st.set_page_config(page_title="AtlasDocs", page_icon="🧠", layout="wide")
    apply_modern_ui()

    st.sidebar.title("AtlasDocs")
    page = st.sidebar.radio("Navigate", ["Landing", "Demo"], index=0)

    if page == "Landing":
        landing_page()
    else:
        demo_page()


if __name__ == "__main__":
    main()
