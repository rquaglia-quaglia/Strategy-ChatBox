# app_assignments.py — Dual-retriever Assignments Chatbot
# Q&A: Scripts (scoped per assignment) + Assignments, NON-SPOILER
# Upload feedback: Assignments only (rubrics/instructor notes)

import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# --- Secrets/env (local .env allowed; Streamlit Cloud overrides via Secrets) ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

def get_secret(name: str, default: str | None = None) -> str | None:
    # Prefer Streamlit secrets, then env
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
CHAT_MODEL     = get_secret("CHAT_MODEL", "gpt-5")
EMBED_MODEL    = get_secret("EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or Settings → Secrets (Cloud).")
    st.stop()

# --- LLM / Vector store bits ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Lightweight file readers for uploads ---
from PyPDF2 import PdfReader
import docx2txt

# ---------------------------- UI CONFIG ----------------------------
st.set_page_config(page_title="Assignments Chatbot", page_icon="📝", layout="wide")
st.title("Assignments Chatbot")
st.caption("Select your assignment, ask non-spoiler questions, upload drafts, and receive feedback against instructor notes.")

INDEX_SCRIPTS = "vectorstore_scripts"
INDEX_ASSIGN  = "vectorstore_assignments"

ASSIGNMENTS = [
    "2.1 — Winning Ambition (Lumora)",
    "3.1 — Playing Field (Lumora)",
    "4.1 — Value Proposition (Lumora)",
    "5.1 — Operating Model (Lumora)",
    "6.1 — Strategic & Enabling Priorities (Lumora)",
]

# Map each assignment to the most relevant script modules
ASSIGNMENT_TO_MODULES = {
    "2.1 — Winning Ambition (Lumora)": ["Module 2", "Module 1"],
    "3.1 — Playing Field (Lumora)":    ["Module 3", "Module 1"],
    "4.1 — Value Proposition (Lumora)":["Module 4", "Module 1", "Module 2"],
    "5.1 — Operating Model (Lumora)":  ["Module 5", "Module 1", "Module 2", "Module 4"],
    "6.1 — Strategic & Enabling Priorities (Lumora)":["Module 6", "Module 1", "Module 5"],
}

# Session state
if "hist_assign" not in st.session_state:
    st.session_state["hist_assign"] = []
if "hints_shown" not in st.session_state:
    st.session_state["hints_shown"] = 0

# ---------------------------- Cache loaders ----------------------------
@st.cache_resource(show_spinner=True)
def load_vs(folder: str, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

if not Path(INDEX_SCRIPTS).exists() or not Path(INDEX_ASSIGN).exists():
    st.error("Missing FAISS indexes. Expected 'vectorstore_scripts' and 'vectorstore_assignments' with index files.")
    st.stop()

vs_scripts = load_vs(INDEX_SCRIPTS, EMBED_MODEL, OPENAI_API_KEY)
vs_assign  = load_vs(INDEX_ASSIGN,  EMBED_MODEL, OPENAI_API_KEY)

# ---------------------------- Non-spoiler guardrails ----------------------------
SOLUTION_SEEKING_PATTERNS = [
    r"\bwhat('?s| is) the (answer|solution)\b",
    r"\bgive me (the )?answer\b",
    r"\bwhat should (we|i) choose\b",
    r"\bwhat('?s| is) (lumora('?s)? )?winning ambition\b",
    r"\bjust tell me\b",
    r"\bwhat('?s| is) the (final|recommended) (choice|strategy)\b",
]
solution_seeking_re = re.compile("|".join(SOLUTION_SEEKING_PATTERNS), re.IGNORECASE)

def is_solution_seeking(text: str) -> bool:
    return bool(solution_seeking_re.search(text or ""))

NON_SPOILER_SYSTEM = (
    "You are a teaching assistant in NON-SPOILER mode.\n"
    "Rules:\n"
    "- NEVER reveal the actual solution, final choices, numeric answers, or the phrasing of the official answer.\n"
    "- Do NOT write the recommended Winning Ambition / Playing Field / Value Proposition / Operating Model / Priorities for Lumora.\n"
    "- Instead: guide with method, structure, rubrics, clarifying questions, and references to pages/sections.\n"
    "- Prefer bullet steps, decision criteria, and examples from the scripts/briefs without giving away the final choice.\n"
    "- If the user explicitly asks for the solution, politely refuse and suggest they upload a draft to get feedback.\n"
)

NON_SPOILER_USER_TEMPLATE = (
    "Assignment: {assignment}\n\n"
    "Student question: {question}\n\n"
    "Use the context below (scripts first, then assignment notes) to provide guidance WITHOUT spoilers.\n\n"
    "Context:\n{context}\n\n"
    "Return:\n"
    "- 2–4 bullet steps to work the task\n"
    "- Key criteria to consider\n"
    "- Where to find the info (cite file names + pages)\n"
    "- 1–2 Socratic questions to push their thinking\n"
)

# ---------------------------- Helpers ----------------------------
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

def nice_src(meta: dict) -> Tuple[str, int | None]:
    p = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
    return Path(p).name, meta.get("page")

def filter_docs_by_modules(docs, allowed_modules: List[str]):
    if not allowed_modules:
        return docs
    keep = []
    for d in docs:
        fname = (d.metadata.get("source") or d.metadata.get("file_path") or "").lower()
        if any(mod.lower() in fname for mod in allowed_modules):
            keep.append(d)
    return keep

def build_context(docs) -> Tuple[str, list]:
    """Return concatenated context text + the doc list (for later source display)."""
    text = "\n\n".join([d.page_content for d in docs])
    return text, docs

# ---------------------------- UI: assignment selection ----------------------------
colA, colB = st.columns([2,1])
with colA:
    selected = st.selectbox("Which assignment are you working on?", ASSIGNMENTS, index=0)
with colB:
    if st.button("Reveal next hint"):
        st.session_state["hints_shown"] += 1

# ---------------------------- Q&A (non-spoiler) ----------------------------
st.subheader("Ask a question")
with st.form("qa"):
    q = st.text_input("Your question about the assignment:", placeholder="e.g., What are the deliverables in 5.1?")
    go = st.form_submit_button("Ask")

if go and q.strip():
    with st.spinner("Thinking..."):
        hist = st.session_state["hist_assign"]
        q_tagged = f"[Assignment: {selected}] {q}"

        if is_solution_seeking(q):
            ans = (
                "I can’t give you the solution directly. Let’s keep this non-spoiler:\n\n"
                "• Review the assignment brief to list the exact deliverables.\n"
                "• Identify relevant frameworks in the scripts and apply them to Lumora’s context.\n"
                "• Draft your reasoning, then upload it here — I’ll give detailed feedback against the instructor notes."
            )
            srcs = []
        else:
            # retrieve from assignments (rubrics/briefs)
            assign_docs = vs_assign.similarity_search(q_tagged, k=3)

            # retrieve from scripts (scoped to relevant modules)
            allowed = ASSIGNMENT_TO_MODULES.get(selected, [])
            scripts_raw = vs_scripts.similarity_search(q_tagged, k=8)
            scripts_scoped = filter_docs_by_modules(scripts_raw, allowed)
            if len(scripts_scoped) < 3:  # graceful fallback
                scripts_scoped = scripts_raw[:3]

            # build context: scripts first (method), then assignments (rubric)
            merged_docs = scripts_scoped + assign_docs
            context, srcs = build_context(merged_docs)

            llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
            user_prompt = NON_SPOILER_USER_TEMPLATE.format(
                assignment=selected,
                question=q,
                context=context
            )
            ans = llm.invoke([
                {"role": "system", "content": NON_SPOILER_SYSTEM},
                {"role": "user", "content": user_prompt}
            ]).content

        hist.extend([("user", q), ("assistant", ans)])

        st.markdown("### Answer (non-spoiler)")
        st.write(ans)

        if srcs:
            st.markdown("### Sources")
            for i, d in enumerate(srcs, 1):
                name, page = nice_src(d.metadata or {})
                st.write(f"{i}. {name}" + (f" — page {page}" if page else ""))

# ---------------------------- Upload & feedback ----------------------------
st.subheader("Upload your draft for feedback")
uploaded = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=False)

def feedback_system(assign_name: str) -> str:
    return (
        f"You are a teaching assistant evaluating a student draft for '{assign_name}'. "
        "Use ONLY the retrieved instructor notes and assignment brief. Provide:\n"
        "1) Completeness vs deliverables\n"
        "2) Strengths & weaknesses\n"
        "3) Gaps or incorrect assumptions\n"
        "4) 3–5 prioritized next steps\n"
        "Cite file names/pages when possible."
    )

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
            # pull focused instructor notes / rubric from assignments index only
            focus_q = f"[Instructor Notes][Assignment: {selected}] rubric, deliverables, evaluation criteria, common pitfalls"
            notes_docs = vs_assign.similarity_search(focus_q, k=5)
            notes_text = "\n\n".join([d.page_content for d in notes_docs])

            llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
            review = llm.invoke([
                {"role": "system", "content": feedback_system(selected)},
                {"role": "user", "content": f"--- STUDENT SUBMISSION ---\n{text[:9000]}\n\n--- INSTRUCTOR NOTES ---\n{notes_text[:5000]}"},
            ]).content

            st.markdown("### Feedback")
            st.write(review)

            if notes_docs:
                st.markdown("### Sources used")
                for i, d in enumerate(notes_docs, 1):
                    name, page = nice_src(d.metadata or {})
                    st.write(f"{i}. {name}" + (f" — page {page}" if page else ""))

# ---------------------------- Hints / progression ----------------------------
if st.session_state["hints_shown"] > 0:
    st.subheader("Hints")
    hints = [
        "Level 1: Re-read the assignment brief and list the deliverables.",
        "Level 2: Connect your reasoning to the Strategy-in-Action Canvas choices.",
        "Level 3: Use evidence from scripts (quote + page) to support your recommendation.",
    ]
    for i in range(min(st.session_state["hints_shown"], len(hints))):
        st.markdown(f"- {hints[i]}")

with st.expander("Chat history"):
    for role, msg in st.session_state["hist_assign"]:
        st.markdown(f"**{'You' if role=='user' else 'Assistant'}:** {msg}")
