# verify_ingestion.py
import os
from pathlib import Path
from collections import defaultdict

# Try loading .env for local use
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY. Put it in .env or set it in your environment.")

# LangChain / FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

ROOT = Path.cwd()

# Configure your folders here (already matching your layout)
SETS = [
    {
        "name": "Scripts",
        "materials_dir": ROOT / "course_materials_scripts",
        "vectorstore_dir": ROOT / "vectorstore_scripts",
    },
    {
        "name": "Assignments",
        "materials_dir": ROOT / "course_materials_assignments",
        "vectorstore_dir": ROOT / "vectorstore_assignments",
    },
]

def base(name: str) -> str:
    """lowercased base filename without extension"""
    return Path(name).stem.lower()

def list_expected_pdfs(materials_dir: Path):
    """All PDFs present in the materials folder"""
    return sorted([p for p in materials_dir.glob("*.pdf")])

def list_ingested_sources(vectorstore_dir: Path):
    """Load FAISS and collect source file names from document metadata"""
    if not vectorstore_dir.exists():
        raise SystemExit(f"Missing vectorstore folder: {vectorstore_dir}")

    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.load_local(
        folder_path=str(vectorstore_dir),
        embeddings=emb,
        allow_dangerous_deserialization=True
    )

    # Pull *all* docs via an index scan: use retriever search on generic terms to collect many docs
    # But better: directly access docstore
    docstore = vs.docstore._dict  # LangChain FAISS docstore keeps docs here
    sources = []
    for _id, doc in docstore.items():
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source") or meta.get("file_path")
        if src:
            sources.append(Path(src).name)
    return sorted(set(sources))

def compare(expected_files, ingested_filenames):
    """Compare expected .pdf files vs ingested source names"""
    exp_bases = {base(p.name): p.name for p in expected_files}
    ing_bases = {base(n): n for n in ingested_filenames}

    found = []
    missing = []
    for k, display in exp_bases.items():
        if k in ing_bases:
            found.append((display, ing_bases[k]))
        else:
            missing.append(display)

    return found, missing, exp_bases, ing_bases

def main():
    for cfg in SETS:
        name = cfg["name"]
        materials = cfg["materials_dir"]
        vdir = cfg["vectorstore_dir"]

        print(f"\n==============================")
        print(f"Checking: {name}")
        print(f"Materials:   {materials}")
        print(f"Vectorstore: {vdir}")

        if not materials.exists():
            print(f"!! Materials folder not found: {materials}")
            continue

        expected = list_expected_pdfs(materials)
        if not expected:
            print("No PDFs found in materials folder.")
            continue

        ingested = list_ingested_sources(vdir)

        found, missing, exp_bases, ing_bases = compare(expected, ingested)

        print(f"\nSummary:")
        print(f"  PDFs in materials: {len(expected)}")
        print(f"  Distinct sources in vectorstore: {len(ing_bases)}")
        print(f"  Found: {len(found)}")
        print(f"  Missing: {len(missing)}")

        if found:
            print("\nFound (materials → ingested source name):")
            for mname, sname in sorted(found):
                print(f"  ✓ {mname}  →  {sname}")

        if missing:
            print("\nMissing (present in materials but NOT in vectorstore):")
            for mname in sorted(missing):
                print(f"  ✗ {mname}")

        # Extra: show any sources in the vectorstore that don't match a material PDF
        extras = [s for s in ingested if base(s) not in exp_bases]
        if extras:
            print("\nExtra sources in vectorstore (not found in materials folder):")
            for s in extras:
                print(f"  • {s}")

    print("\nDone.")
    
if __name__ == "__main__":
    main()
