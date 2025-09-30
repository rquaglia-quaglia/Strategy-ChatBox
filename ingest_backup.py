# ingest.py — PDF loader with optional OCR fallback and multi-index support
import os, json, hashlib, argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

def sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def is_truthy(x: str) -> bool:
    return str(x).lower() in {"1","true","yes","y","on"}

def ocr_page_to_text(page, dpi=300, lang="eng"):
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "OCR requested but Pillow/pytesseract not installed. "
            "Install with: pip install pillow pytesseract"
        ) from e
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def extract_pages_with_fallback(pdf_path: Path, use_ocr: bool, ocr_lang: str):
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text").strip()
            if use_ocr and len(txt) < 50:
                try:
                    ocr_txt = ocr_page_to_text(page, dpi=300, lang=ocr_lang).strip()
                    if len(ocr_txt) > len(txt):
                        yield (i + 1, ocr_txt, True)
                        continue
                except Exception as e:
                    print(f"[WARN] OCR failed on {pdf_path.name} p.{i+1}: {e}")
            yield (i + 1, txt, False)

def load_documents(input_dir: Path, use_ocr: bool, ocr_lang: str) -> List[Document]:
    docs: List[Document] = []
    for path in input_dir.rglob("*.pdf"):
        try:
            for page_num, content, ocr_applied in extract_pages_with_fallback(path, use_ocr, ocr_lang):
                if not content:
                    continue
                meta = {
                    "source": str(path),
                    "filename": path.name,
                    "ext": ".pdf",
                    "page": page_num,
                    "ocr_applied": ocr_applied,
                }
                docs.append(Document(page_content=content, metadata=meta))
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)

def load_hash_db(hash_path: Path):
    if hash_path.exists():
        return json.loads(hash_path.read_text())
    return {"page_hashes": {}}

def save_hash_db(hash_path: Path, db: dict):
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(json.dumps(db, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs with optional OCR fallback.")
    parser.add_argument("--input_dir", default="course_materials_scripts", help="Folder with PDFs to ingest")
    parser.add_argument("--output_dir", default="vectorstore_scripts", help="Folder to save FAISS index")
    parser.add_argument("--use_ocr", default=os.getenv("USE_OCR", "0"), help="1/true to enable OCR fallback")
    parser.add_argument("--ocr_lang", default=os.getenv("OCR_LANG", "eng"), help="Tesseract language, e.g. 'eng' or 'eng+ita'")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_ocr    = is_truthy(args.use_ocr)
    ocr_lang   = args.ocr_lang

    VECTOR_DIR = output_dir
    HASH_DB    = VECTOR_DIR / "dochashes.json"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vs = None
    if (VECTOR_DIR / "index.faiss").exists():
        vs = FAISS.load_local(
            str(VECTOR_DIR),
            OpenAIEmbeddings(model=os.getenv("EMBED_MODEL","text-embedding-3-large")),
            allow_dangerous_deserialization=True
        )

    print(f"Ingesting from '{input_dir}' → '{output_dir}'  (OCR={'ON' if use_ocr else 'OFF'}  lang={ocr_lang})")
    raw_pages = load_documents(input_dir, use_ocr, ocr_lang)
    print(f"Loaded {len(raw_pages)} pages (pre-split).")

    chunks = split_docs(raw_pages)
    print(f"Produced {len(chunks)} chunks.")

    hash_db = load_hash_db(HASH_DB)
    page_hashes = hash_db["page_hashes"]

    to_add = []
    for d in tqdm(chunks, desc="Dedup filtering"):
        meta_bits = f"{d.metadata.get('source','')}|{d.metadata.get('page','')}|{d.metadata.get('filename','')}"
        h = sha256(d.page_content + "||" + meta_bits)
        if h not in page_hashes:
            page_hashes[h] = {"source": d.metadata.get("source",""), "filename": d.metadata.get("filename","")}
            to_add.append(d)

    if not to_add:
        print("Nothing new to add. Index is up to date.")
        return

    print(f"Embedding {len(to_add)} new chunks…")
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL","text-embedding-3-large"))
    if vs is None:
        vs = FAISS.from_documents(to_add, embeddings)
    else:
        vs.add_documents(to_add, embeddings=embeddings)

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VECTOR_DIR))
    save_hash_db(HASH_DB, hash_db)
    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
