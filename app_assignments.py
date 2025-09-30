# app_assignments.py — Assignments Chatbot (Non-spoiler Q&A + uploads & feedback)

import os
from io import BytesIO
from pathlib import Path
import re
import streamlit as st

# --- Secrets/env (local .env, override with Streamlit Secrets when present) ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if "CHAT_MODEL" in st.secrets:
        CHAT_MODEL = st.secrets["CHAT_MODEL"]
    if "EMBED_MODEL" in st.secrets:
        EMBED_MODEL = st.secrets["EMBED_MODEL"]
except Exception:
    pass

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or Secrets (Cloud).")
    st.stop()

# --- LangChain / OpenAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# --- Helpers for uploads ---
from PyPDF2 import PdfReader
import docx2txt

# --- UI ---
st.set_page_config(page_title="Assignments Chatbot", page_icon="📝", layout="wide")
st.title("Assignments Chatbot")
st.caption("Select your assignment, ask non-spoiler questions, upload drafts, and receive feedback against instructor notes.")

INDEX_FOLDER = "vectorstore_assignments"
ASSIGNMENTS = [
    "2.1 — Winning Ambition (Lumora)",
    "3.1 — Playing Field (Lumora)",
    "4.1 — Value Proposition (Lumora)",
    "5.1 — Operating Model (Lumora)",
    "6.1 — Strategic & Enabling Priorities (Lumora)"
]

if "hist_assign" not in st.session_state:
    st.session_state["hist_assign"] = []
if "hints_shown" not in st.session_state:
    st.session_state["hints_shown"] = 0

# ---- Basic file readers ----
def read_pdf(file_bytes: BytesIO) -> str:
    reader = PdfReader(file_bytes)
    return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()

def read_docx(file_bytes: BytesIO) -> str:
    tmp = Path("._tmp_upload.docx")
    tmp.write_bytes(file_bytes.getbuffer())
    try:
        return docx2txt.process(str(tmp)) or ""
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass

def read_txt(file_bytes: BytesIO) -> str:
    return file_bytes.getvalue().decode("utf-8", errors="ignore")

# ---- Load vectorstore ----
@st.cache_resource(show_spinner=True)
def load_vectorstore(folder_path: str, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

# ---------------- Non-spoiler guardrails ----------------
NON_SPOILER_PROMPT = PromptTemplate.from_template(
    """
You are a teaching assistant in NON-SPOILER mode.

Rules:
- NEVER reveal the actual solution, final choices, numeric answers, or the phrasing of the official answer.
- Do NOT write the recommended Winning Ambition / Playing Field / Value Proposition / Operating Model / Priorities for Lumora.
- Instead: guide with method, structure, rubrics, clarifying questions, and references to pages/sections.
- Prefer bullet steps, decision criteria, and examples from the scripts/briefs without giving away the final choice.
- If the user explicitly asks for the solution, politely refuse and suggest they upload a draft to get feedback.

Student question: {question}

Use the context below to provide guidance without spoilers.

Context:
{context}

Return:
- 2–4 bullet steps to work the task
- Key criteria to consider
- Where to find the info (cite file names + pages)
- 1–2 Socratic questions to push their thinking
"""
)

SOLUTION_SEEKING_PATTERNS = [
    r"\bwhat('?s| is) the (answer|solution)\b",
    r"\bgive me (the )?answer\b",
    r"\bwhat should (we|i) choose\b",
    r"\bwhat('?s| is) the winning ambition\b",
    r"\bwhat('?s| is) our (final|recommended) (choice|strategy)\b",
    r"\bjust tell me\b",
]
solution_seeking_re = re.compile("|".join(SOLUTION_SEEKING_PATTERNS), re.IGNORECASE)

def is_solution_seeking(text: str) -> bool:
    return bool(solution_seeking_re.search(text or ""))

# Build a ConversationRetrievalChain with a non-spoiler answer prompt
def build_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": NON_SPOILER_PROMPT},
        verbose=False,
    )
    return chain

# Guard
if not Path(INDEX_FOLDER).exists():
    st.error(f"Missing '{INDEX_FOLDER}'. Ensure index.faiss/index.pkl exist.")
    st.stop()

vs = load_vectorstore(INDEX_FOLDER, EMBED_MODEL, OPENAI_API_KEY)
chain = build_chain(vs)

# ---- UI: Assignment selection ----
colA, colB = st.columns([2,1])
with colA:
    selected = st.selectbox("Which assignment are you working on?", ASSIGNMENTS, index=0)
with colB:
    if st.button("Reveal next hint"):
        st.session_state["hints_shown"] += 1

# ---- Chat Q&A ----
st.subheader("Ask a question")
with st.form("qa"):
    q = st.text_input("Your question about the assignment:", placeholder="e.g., What are the deliverables in 5.1?")
    go = st.form_submit_button("Ask")

if go and q.strip():
    with st.spinner("Thinking..."):
        hist = st.session_state["hist_assign"]
        # Bias retrieval with the selected assignment
        q_tagged = f"[Assignment: {selected}] {q}"

        # If the student asks for the solution, short-circuit to a refusal + guidance
        if is_solution_seeking(q):
            ans = (
                "I can’t give you the solution directly. Let’s keep this non-spoiler:\n\n"
                "• Review the assignment brief and clarify the decision you must make.\n"
                "• Identify the evidence you need (from the case/scripts) to justify that decision.\n"
                "• Draft your reasoning, then upload it here — I’ll give detailed feedback against the instructor notes.\n\n"
                "Tip: look for the criteria and examples mentioned in the brief and scripts, not the final choice."
            )
            srcs = []
        else:
            out = chain({"question": q_tagged, "chat_history": hist})
            ans = out.get("answer", "")
            srcs = out.get("source_documents", [])

        hist.extend([("user", q), ("assistant", ans)])
        st.markdown("### Answer (non-spoiler)")
        st.write(ans)

        if srcs:
            st.markdown("### Sources")
            for i, d in enumerate(srcs, 1):
                meta = d.metadata or {}
                src = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
                page = meta.get("page")
                st.write(f"{i}. {Path(src).name}" + (f" — page {page}" if page is not None else ""))

# ---- Upload & feedback ----
st.subheader("Upload your draft for feedback")
uploaded = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf","docx","txt"], accept_multiple_files=False)

def instructor_feedback_prompt(assign_name: str) -> str:
    return f"""
You are a teaching assistant evaluating a student draft for '{assign_name}' (Lumora).
Use ONLY the retrieved instructor notes and assignment brief to:
1) assess completeness vs required deliverables,
2) highlight strengths/weaknesses,
3) point out gaps or incorrect assumptions,
4) give 3–5 prioritized next steps,
5) cite the specific sources/pages you used.

IMPORTANT: Here you may reference the correct direction if needed, but keep the tone coaching and actionable.
"""

if uploaded is not None and st.button("Analyze submission"):
    with st.spinner("Evaluating your submission..."):
        ext = uploaded.name.lower().split(".")[-1]
        try:
            if ext == "pdf":
                text = read_pdf(BytesIO(uploaded.read()))
            elif ext == "docx":
                text = read_docx(BytesIO(uploaded.read()))
            else:
                text = read_txt(BytesIO(uploaded.read()))
        except Exception as e:
            st.error(f"Could not read file: {e}")
            text = ""

        if text.strip():
            # Retrieve instructor notes focussed on the chosen assignment
            focus_q = f"[Instructor Notes][Assignment: {selected}] rubric, deliverables, evaluation criteria, common pitfalls"
            out = chain({"question": focus_q, "chat_history": []})
            notes = out.get("answer", "")
            srcs = out.get("source_documents", [])
            llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)

            system = instructor_feedback_prompt(selected)
            user_msg = f"--- STUDENT SUBMISSION ---\n{text[:9000]}\n\n--- INSTRUCTOR NOTES (retrieved) ---\n{notes[:5000]}"
            review = llm.invoke([
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ]).content

            st.markdown("### Feedback")
            st.write(review)

            if srcs:
                st.markdown("### Sources used")
                for i, d in enumerate(srcs, 1):
                    meta = d.metadata or {}
                    src = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
                    page = meta.get("page")
                    st.write(f"{i}. {Path(src).name}" + (f" — page {page}" if page is not None else ""))

# ---- Hints / progression ----
if st.session_state["hints_shown"] > 0:
    st.subheader("Hints")
    hints = [
        "Level 1: Re-read the assignment brief and list the exact deliverables.",
        "Level 2: Connect your reasoning to the Strategy-in-Action Canvas choices from earlier modules.",
        "Level 3: Support your case using quotes/data/examples from the scripts (cite pages)."
    ]
    for i in range(min(st.session_state["hints_shown"], len(hints))):
        st.markdown(f"- {hints[i]}")

with st.expander("Chat history"):
    for role, msg in st.session_state["hist_assign"]:
        st.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {msg}")
