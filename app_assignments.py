# app_assignments.py — Assignments Chatbot (non-spoiler) with dual retrieval
# Scripts → methods/frameworks/theory
# Assignments → deliverables/case/context/rubric/how-to
# Q&A uses both corpora; upload feedback uses assignments only

import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# -------------------- Secrets / env --------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

def get_secret(name: str, default: str | None = None) -> str | None:
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

# -------------------- LLM / Vector store --------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------- Upload readers --------------------
from PyPDF2 import PdfReader
import docx2txt

# -------------------- UI config --------------------
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

# Map each assignment to the most relevant script modules (loose scoping + fallback)
ASSIGNMENT_TO_MODULES = {
    "2.1 — Winning Ambition (Lumora)": ["Module 2", "Module 1"],
    "3.1 — Playing Field (Lumora)":    ["Module 3", "Module 1"],
    "4.1 — Value Proposition (Lumora)":["Module 4", "Module 1", "Module 2"],
    "5.1 — Operating Model (Lumora)":  ["Module 5", "Module 1", "Module 2", "Module 4"],
    "6.1 — Strategic & Enabling Priorities (Lumora)":["Module 6", "Module 1", "Module 5"],
}

# -------------------- Session state --------------------
if "hist_assign" not in st.session_state:
    st.session_state["hist_assign"] = []
if "hints_shown" not in st.session_state:
    st.session_state["hints_shown"] = 0

# -------------------- Cache loaders --------------------
@st.cache_resource(show_spinner=True)
def load_vs(folder: str, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)

if not Path(INDEX_SCRIPTS).exists() or not Path(INDEX_ASSIGN).exists():
    st.error("Missing FAISS indexes. Expected 'vectorstore_scripts' and 'vectorstore_assignments' with index files.")
    st.stop()

vs_scripts = load_vs(INDEX_SCRIPTS, EMBED_MODEL, OPENAI_API_KEY)
vs_assign  = load_vs(INDEX_ASSIGN,  EMBED_MODEL, OPENAI_API_KEY)

# -------------------- Non-spoiler guardrails --------------------
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
    "Use sources as follows:\n"
    "- Scripts (course modules) → the ONLY source for methods, frameworks, and theory.\n"
    "- Assignments (briefs/handbooks/rubrics) → the ONLY source for deliverables, case context, grading criteria, and how to execute the exercise.\n"
    "Rules:\n"
    "- NEVER reveal the official solution/recommendation.\n"
    "- Prefer numbered steps and decision criteria; ask 1–2 clarifying questions.\n"
    "- Cite where info comes from: label each citation as [Script] or [Assignment] with filename and page.\n"
)

NON_SPOILER_USER_TEMPLATE = (
    "Assignment: {assignment}\n\n"
    "Student question: {question}\n\n"
    "Context for METHODS & THEORY (from Scripts):\n{scripts_context}\n\n"
    "Context for DELIVERABLES & CASE (from Assignments):\n{assign_context}\n\n"
    "Return (no spoilers):\n"
    "1) Method & theory (from scripts only):\n"
    "2) Deliverables & case pointers (from assignments only):\n"
    "3) 2–4 steps to progress\n"
    "4) 1–2 Socratic questions\n"
    "Include short citations like [Script: <file> p.X] or [Assignment: <file> p.Y].\n"
)

# -------------------- Helpers --------------------
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

# -------------------- UI: assignment selection --------------------
colA, colB = st.columns([2,1])
with colA:
    selected = st.selectbox("Which assignment are you working on?", ASSIGNMENTS, index=0)
with colB:
    if st.button("Reveal next hint"):
        st.session_state["hints_shown"] += 1

# -------------------- Q&A (non-spoiler) --------------------
st.subheader("Ask a question")
with st.form("qa"):
    q = st.text_input(
        "Your question about the assignment:",
        placeholder="e.g., How to design metrics that inform Playing Field and Strategic Priorities?"
    )
    go = st.form_submit_button("Ask")

if go and q.strip():
    with st.spinner("Thinking..."):
        st.session_state["hist_assign"].append(("user", q))
        q_tagged = f"[Assignment: {selected}] {q}"

        # If asking for the solution, refuse politely
        if is_solution_seeking(q):
            ans = (
                "I can’t give you the solution directly. Let’s keep this non-spoiler:\n\n"
                "• Use the assignment brief to list deliverables and clarify scope.\n"
                "• Use the scripts to pick the right frameworks and structure your reasoning.\n"
                "• Draft your approach and upload it here — I’ll give feedback against the instructor notes."
            )
            srcs_scripts, srcs_assign = [], []

        else:
            # --- Assignments: deliverables/case (always include a few) ---
            assign_docs = vs_assign.similarity_search(q_tagged, k=2)

            # --- Scripts: methods/frameworks (ensure we ALWAYS include enough scripts) ---
            allowed = ASSIGNMENT_TO_MODULES.get(selected, [])
            scripts_raw = vs_scripts.similarity_search(q_tagged, k=10)
            scripts_scoped = filter_docs_by_modules(scripts_raw, allowed)
            if len(scripts_scoped) < 4:
                extra = [d for d in scripts_raw if d not in scripts_scoped]
                scripts_scoped += extra[: max(0, 4 - len(scripts_scoped))]
            scripts_scoped = scripts_scoped[:4]

            # Build split contexts
            scripts_context = "\n\n".join([d.page_content for d in scripts_scoped])
            assign_context  = "\n\n".join([d.page_content for d in assign_docs])

            llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
            user_prompt = NON_SPOILER_USER_TEMPLATE.format(
                assignment=selected,
                question=q,
                scripts_context=scripts_context,
                assign_context=assign_context
            )
            ans = llm.invoke([
                {"role": "system", "content": NON_SPOILER_SYSTEM},
                {"role": "user", "content": user_prompt}
            ]).content

            srcs_scripts, srcs_assign = scripts_scoped, assign_docs

        st.session_state["hist_assign"].append(("assistant", ans))

        st.markdown("### Answer (non-spoiler)")
        st.write(ans)

        # Split sources by corpus
        st.markdown("### Sources")
        if srcs_scripts:
            st.markdown("**From scripts (methods & theory):**")
            for i, d in enumerate(srcs_scripts, 1):
                name, page = nice_src(d.metadata or {})
                st.write(f"{i}. {name}" + (f" — page {page}" if page else ""))

        if srcs_assign:
            st.markdown("**From assignments (deliverables & case):**")
            for i, d in enumerate(srcs_assign, 1):
                name, page = nice_src(d.metadata or {})
                st.write(f"{i}. {name}" + (f" — page {page}" if page else ""))

        if not srcs_scripts and not srcs_assign:
            st.write("_No sources retrieved._")

# -------------------- Upload & feedback (assignments only) --------------------
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

def read_any(upload) -> str:
    ext = upload.name.lower().split(".")[-1]
    data = upload.read()
    if ext == "pdf":
        return read_pdf(BytesIO(data))
    if ext == "docx":
        return read_docx(BytesIO(data))
    return data.decode("utf-8", errors="ignore")

if uploaded is not None and st.button("Analyze submission"):
    with st.spinner("Evaluating your submission..."):
        try:
            text = read_any(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            text = ""

        if text.strip():
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

# -------------------- Hints / progression --------------------
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
