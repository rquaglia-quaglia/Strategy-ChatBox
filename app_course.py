# app_course.py  — Course-only Streamlit chatbot (local .env + Cloud secrets)

import os
from pathlib import Path
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
except Exception:
    # No secrets.toml locally — ignore
    pass

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or to Secrets (Cloud).")
    st.stop()

# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# --- UI ---
st.set_page_config(page_title="Course Chatbot", page_icon="🎓", layout="wide")
st.title("Course Chatbot")
st.caption("Ask questions about the **course scripts** and get sourced answers.")

INDEX_FOLDER = "vectorstore_scripts"

# Separate chat history for this bot
if "hist_course" not in st.session_state:
    st.session_state["hist_course"] = []

@st.cache_resource(show_spinner=True)
def load_vectorstore(folder_path: str, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

def build_chain():
    vs = load_vectorstore(INDEX_FOLDER, EMBED_MODEL, OPENAI_API_KEY)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

# Guard: index must exist
if not Path(INDEX_FOLDER).exists():
    st.error(f"Missing '{INDEX_FOLDER}'. Ensure index.faiss and index.pkl are present.")
    st.stop()

chain = build_chain()

with st.form("qa"):
    q = st.text_input(
        "Your question about the course:",
        placeholder="e.g., Summarize the Winning Ambition concept."
    )
    go = st.form_submit_button("Ask")

if go and q.strip():
    with st.spinner("Thinking..."):
        hist = st.session_state["hist_course"]
        out = chain({"question": q, "chat_history": hist})
        ans = out.get("answer", "")
        srcs = out.get("source_documents", [])

        # update history
        hist.extend([("user", q), ("assistant", ans)])

        st.markdown("### Answer")
        st.write(ans)

        if srcs:
            st.markdown("### Sources")
            for i, d in enumerate(srcs, 1):
                meta = d.metadata or {}
                src = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
                page = meta.get("page")
                name = Path(src).name
                st.write(f"{i}. {name}" + (f" — page {page}" if page is not None else ""))

with st.expander("Chat history"):
    for role, msg in st.session_state["hist_course"]:
        st.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {msg}")
