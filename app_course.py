# app_course.py  — Course-only Streamlit chatbot (modern RAG, admin-only debug)
# Answers questions using ONLY the course scripts vectorstore

import os
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# --- Secrets/env (local .env, override with Streamlit Secrets when present) ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # let .env win over stale system vars
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

try:
    # On Streamlit Cloud, st.secrets exists — override env values
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if "CHAT_MODEL" in st.secrets:
        CHAT_MODEL = st.secrets["CHAT_MODEL"]
    if "EMBED_MODEL" in st.secrets:
        EMBED_MODEL = st.secrets["EMBED_MODEL"]
    DEBUG_MODE = str(st.secrets.get("DEBUG_MODE", "false")).strip().lower() in {"1","true","yes","on"}
except Exception:
    DEBUG_MODE = False  # no secrets.toml locally — default off

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or to Secrets (Cloud).")
    st.stop()

# --- LangChain / OpenAI (modern RAG pieces) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- UI ---
st.set_page_config(page_title="Course Chatbot", page_icon="🎓", layout="wide")
st.title("Course Chatbot")
st.caption("Ask questions about the **course scripts** (methods, frameworks, theory).")

# --- Admin-only debug (hidden from students unless DEBUG_MODE=true) ---
IS_LOCAL = os.getenv("LOCAL_DEV", "0") == "1"
DEBUG_ALLOWED = IS_LOCAL or DEBUG_MODE

if DEBUG_ALLOWED:
    DEBUG = st.sidebar.checkbox("Debug retrieval", value=False)
    if DEBUG:
        st.sidebar.markdown("🔒 **Staff debug active**")
else:
    DEBUG = False

def debug_log(msg: str, data: Any = None):
    if DEBUG:
        with st.sidebar.expander("Debug log", expanded=True):
            st.write(msg)
            if data is not None:
                try:
                    st.json(data)
                except Exception:
                    st.write(data)

INDEX_FOLDER = "vectorstore_scripts"

# Separate chat history for this bot (for display only; not used by the chain)
if "hist_course" not in st.session_state:
    st.session_state["hist_course"] = []

@st.cache_resource(show_spinner=True)
def load_vectorstore(folder_path: str, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    try:
        return FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        ), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# Guard: index must exist
if not Path(INDEX_FOLDER).exists():
    st.error(f"Missing '{INDEX_FOLDER}'. Ensure index.faiss and index.pkl are present.")
    st.stop()

vs_scripts, vs_err = load_vectorstore(INDEX_FOLDER, EMBED_MODEL, OPENAI_API_KEY)
if vs_err:
    st.error("❌ Failed to load vectorstore_scripts")
    debug_log("scripts_vs load error", {"error": vs_err})

# Build retriever & LLM
retriever = vs_scripts.as_retriever(search_kwargs={"k": 5}) if vs_scripts else None
llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)

# Prompt for RAG
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful teaching assistant for the course scripts. "
     "Answer using ONLY the provided context from the scripts. "
     "If the answer is not in the context, say you don’t have enough info."),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

def format_docs(docs):
    return "\n\n---\n".join(d.page_content for d in docs)

# Modern RAG chain: (question) -> retrieve -> format -> prompt -> LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # pass q through; use retriever for context
    | prompt
    | llm
) if retriever else None

# Sidebar self-test (only when debug allowed)
if DEBUG_ALLOWED and st.sidebar.button("Run retrieval self-test"):
    info = {
        "scripts_dir_exists": Path(INDEX_FOLDER).exists(),
        "scripts_loaded": vs_scripts is not None,
        "scripts_error": vs_err,
    }
    if retriever:
        try:
            test_q = "What is the Play-to-Win framework?"
            hits = retriever.get_relevant_documents(test_q)
            info["hit_count"] = len(hits)
            info["example_source"] = (hits[0].metadata or {}).get("source","") if hits else None
        except Exception as e:
            info["probe_error"] = str(e)
    st.success("Self-test complete. See sidebar for details.")
    debug_log("Self-test results", info)

# --- Q&A form ---
with st.form("qa"):
    q = st.text_input(
        "Your question about the course:",
        placeholder="e.g., Summarize the Winning Ambition concept."
    )
    go = st.form_submit_button("Ask")

if go and q.strip():
    if not rag_chain:
        st.error("Vectorstore not loaded; cannot answer right now.")
    else:
        with st.spinner("Thinking..."):
            # Retrieve docs (for sources panel)
            docs = retriever.get_relevant_documents(q)

            # Generate answer
            answer = rag_chain.invoke(q).content

            # Update history (display-only)
            st.session_state["hist_course"].extend([("user", q), ("assistant", answer)])

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources (scripts)")
            if docs:
                for i, d in enumerate(docs, 1):
                    meta = d.metadata or {}
                    src = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
                    page = meta.get("page")
                    name = Path(src).name
                    st.write(f"{i}. {name}" + (f" — page {page}" if page is not None else ""))
            else:
                st.caption("_(no script sources retrieved)_")

            if DEBUG:
                rows = []
                for rank, d in enumerate(docs, start=1):
                    md = d.metadata or {}
                    rows.append({
                        "rank": rank,
                        "source": md.get("source", ""),
                        "page": md.get("page", ""),
                        "module": md.get("module", ""),
                    })
                st.subheader("Retrieval debug")
                st.dataframe(rows, use_container_width=True)

# --- History panel ---
with st.expander("Chat history"):
    for role, msg in st.session_state["hist_course"]:
        st.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {msg}")
