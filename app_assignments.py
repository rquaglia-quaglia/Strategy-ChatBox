# app_assignments.py — Assignments Chatbot (non-spoiler) with dual retrieval & admin-only debug
# Scripts → methods/frameworks/theory
# Assignments → deliverables/case/context/rubric/how-to
# Q&A uses both corpora; upload feedback uses assignments only

import os
import re
import json
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st

# -------------------- Secrets / env --------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
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

# --- Admin-only debug controls (hidden from students) ---
def str2bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "on"}

IS_LOCAL = os.getenv("LOCAL_DEV", "0") == "1"                         # optional: set in .env locally
ALLOW_DEBUG = str2bool(get_secret("DEBUG_MODE", "false") or "false")  # set in Streamlit Cloud Secrets
DEBUG_ALLOWED = IS_LOCAL or ALLOW_DEBUG

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

# -------------------- Paths / Indexes --------------------
BASE_DIR = Path(__file__).parent.resolve()
INDEX_SCRIPTS = BASE_DIR / "vectorstore_scripts"
INDEX_ASSIGN  = BASE_DIR / "vectorstore_assignments"

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
def load_vs(folder: Path, embed_model: str, api_key: str):
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)
    try:
        vs = FAISS.load_local(
            str(folder),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vs, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

if not INDEX_SCRIPTS.exists() or not INDEX_ASSIGN.exists():
    st.error("Missing FAISS indexes. Expected 'vectorstore_scripts' and 'vectorstore_assignments' with index files.")
    st.stop()

vs_scripts, scripts_err = load_vs(INDEX_SCRIPTS, EMBED_MODEL, OPENAI_API_KEY)
vs_assign,  assign_err  = load_vs(INDEX_ASSIGN,  EMBED_MODEL, OPENAI_API_KEY)

if assign_err:
    st.error("❌ Failed to load vectorstore_assignments")
    debug_log("assign_vs load error", {"error": assign_err, "trace": traceback.format_exc()})

if scripts_err:
    st.warning("⚠️ Failed to load vectorstore_scripts — answers will work but script sources may be missing.")
    debug_log("scripts_vs load error", {"error": scripts_err, "trace": traceback.format_exc()})

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

def nice_src(meta: dict) -> Tuple[str, Optional[int]]:
    p = meta.get("source") or meta.get("file_path") or meta.get("filename") or "Unknown"
    return Path(p).name, meta.get("page")

# --- Smarter module matching (tolerant to metadata/path differences) ---
def normalize_module_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return label.lower().replace("module", "").strip()

def module_matches(md: Dict[str, Any], allowed_modules: List[str]) -> bool:
    if not allowed_modules:
        return True
    hay = " ".join([
        str(md.get("module","")),
        str(md.get("section","")),
        str(md.get("title","")),
        str(md.get("source","")),
        str(md.get("file_path","")),
        str(md.get("filename","")),
    ]).lower()
    for m in allowed_modules:
        w = normalize_module_label(m)
        if w and w in hay:
            return True
    return False

def tag_corpus(meta: dict) -> str:
    src = (meta.get("source") or meta.get("file_path") or "").lower()
    if "vectorstore_scripts" in src or "course_materials_scripts" in src:
        return "scripts"
    if "vectorstore_assignments" in src or "course_materials_assignments" in src:
        return "assignments"
    return meta.get("corpus", "unknown")

# --- Dual retrieval: always bring in some scripts (with scoped-first logic) ---
def dual_retrieve(
    query: str,
    allowed_modules: List[str],
    k_total: int = 8,
    k_assign: int = 3,
    k_scripts_target: int = 5
):
    results: List[Tuple[Any, float]] = []

    # 1) Assignments (deliverables/case). Prefer scoped (by filename/path) but fallback to unscoped.
    if vs_assign:
        try:
            a = vs_assign.similarity_search_with_score(query, k=max(k_assign, 3))
            a_scoped = [(d,s) for (d,s) in a if module_matches(d.metadata or {}, allowed_modules)]
            results.extend(a_scoped or a[:k_assign])
        except Exception as e:
            debug_log("assign retrieval error", {"error": str(e)})

    # 2) Scripts (methods/frameworks). Try scoped first; if empty, fallback to unscoped.
    scripts_added = 0
    if vs_scripts:
        try:
            s = vs_scripts.similarity_search_with_score(query, k=max(k_scripts_target*2, 8))
            s_scoped = [(d,scr) for (d,scr) in s if module_matches(d.metadata or {}, allowed_modules)]
            pool = s_scoped or s
            for pair in pool:
                results.append(pair)
                scripts_added += 1
                if scripts_added >= k_scripts_target:
                    break
        except Exception as e:
            debug_log("scripts retrieval error", {"error": str(e)})

    # 3) Deduplicate by (source,page) and limit to k_total overall
    seen = set()
    deduped: List[Tuple[Any, float]] = []
    for d,score in results:
        key = ( (d.metadata or {}).get("source"), (d.metadata or {}).get("page") )
        if key not in seen:
            seen.add(key)
            deduped.append((d,score))
        if len(deduped) >= k_total:
            break

    # Split by corpus for downstream use
    scripts_docs = [d for (d,_) in deduped if tag_corpus(d.metadata or {}) == "scripts"]
    assign_docs  = [d for (d,_) in deduped if tag_corpus(d.metadata or {}) == "assignments"]

    return deduped, scripts_docs, assign_docs

def render_sources_grouped(scripts_docs: List[Any], assign_docs: List[Any]):
    st.markdown("### Sources")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**From scripts (methods & theory):**")
        if scripts_docs:
            for i, d in enumerate(scripts_docs, 1):
                name, page = nice_src(d.metadata or {})
                st.write(f"{i}. {name}" + (f" — page {page}" if page is not None else ""))
        else:
            st.caption("_(none)_")
    with cols[1]:
        st.markdown("**From assignments (deliverables & case):**")
        if assign_docs:
            for i, d in enumerate(assign_docs, 1):
                name, page = nice_src(d.metadata or {})
                st.write(f"{i}. {name}" + (f" — page {page}" if page is not None else ""))
        else:
            st.caption("_(none)_")

def show_debug_table(pairs: List[Tuple[Any,float]], header: str = "Retrieval debug (ranked hits)"):
    if not DEBUG or not pairs:
        return
    rows = []
    for rank,(d,score) in enumerate(pairs, start=1):
        md = d.metadata or {}
        rows.append({
            "rank": rank,
            "corpus": tag_corpus(md),
            "score": float(score),
            "module": md.get("module",""),
            "source": md.get("source",""),
            "page": md.get("page",""),
        })
    st.subheader(header)
    st.dataframe(rows, use_container_width=True)

# -------------------- Sidebar: self-test (only when debug allowed) --------------------
if DEBUG_ALLOWED and st.sidebar.button("Run retrieval self-test"):
    info = {
        "scripts_dir_exists": INDEX_SCRIPTS.exists(),
        "assign_dir_exists": INDEX_ASSIGN.exists(),
        "scripts_loaded": vs_scripts is not None,
        "assign_loaded":  vs_assign  is not None,
        "scripts_error": scripts_err,
        "assign_error":  assign_err,
    }
    test_q = "What is the Play-to-Win framework?"
    if vs_assign:
        try:
            a_docs = vs_assign.similarity_search(test_q, k=3)
            info["assign_hits"] = len(a_docs)
            info["assign_example_source"] = (a_docs[0].metadata or {}).get("source","") if a_docs else None
        except Exception as e:
            info["assign_probe_error"] = str(e)
    if vs_scripts:
        try:
            s_docs = vs_scripts.similarity_search(test_q, k=3)
            info["scripts_hits"] = len(s_docs)
            info["scripts_example_source"] = (s_docs[0].metadata or {}).get("source","") if s_docs else None
        except Exception as e:
            info["scripts_probe_error"] = str(e)

    st.success("Self-test complete. See sidebar for details.")
    debug_log("Self-test results", info)

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
            scripts_docs, assign_docs, retrieved_pairs = [], [], []

        else:
            allowed = ASSIGNMENT_TO_MODULES.get(selected, [])
            retrieved_pairs, scripts_docs, assign_docs = dual_retrieve(
                query=q_tagged,
                allowed_modules=allowed,
                k_total=8,
                k_assign=3,
                k_scripts_target=5
            )

            # Build split contexts (cap lengths just to be safe)
            scripts_context = "\n\n".join([d.page_content for d in scripts_docs])[:6000]
            assign_context  = "\n\n".join([d.page_content for d in assign_docs])[:4000]

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

        st.session_state["hist_assign"].append(("assistant", ans))

        st.markdown("### Answer (non-spoiler)")
        st.write(ans)

        # Sources panel (grouped) or guardrail note
        if not is_solution_seeking(q):
            render_sources_grouped(scripts_docs, assign_docs)
            show_debug_table(retrieved_pairs, header="Retrieval debug (ranked hits)")
        else:
            st.markdown("### Sources")
            st.write("_No sources for solution-seeking prompts (guardrail)._")

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
            try:
                notes_docs = vs_assign.similarity_search(focus_q, k=5) if vs_assign else []
            except Exception as e:
                notes_docs = []
                debug_log("assignment notes retrieval error", {"error": str(e)})

            notes_text = "\n\n".join([d.page_content for d in notes_docs])[:6000]

            llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.0)
            review = llm.invoke([
                {"role": "system", "content": feedback_system(selected)},
                {"role": "user", "content": f"--- STUDENT SUBMISSION ---\n{text[:9000]}\n\n--- INSTRUCTOR NOTES ---\n{notes_text}"},
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
