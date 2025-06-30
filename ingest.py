import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs_path = "docs"
all_docs = []

for filename in os.listdir(docs_path):
    full_path = os.path.join(docs_path, filename)
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(full_path)
    elif filename.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(full_path)
    elif filename.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(full_path)
    else:
        continue

    print(f"Loading {filename}...")
    documents = loader.load()
    all_docs.extend(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(all_docs)

print(f"Split into {len(chunks)} chunks. Creating vectorstore...")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")

print("✅ Vectorstore created and saved.")
