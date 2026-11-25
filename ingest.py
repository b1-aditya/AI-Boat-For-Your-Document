"""
ingest.py
---------
Reads uploaded documents (PDF, DOCX, TXT, JSON), splits them into chunks,
and builds/updates two indexes for Hybrid RAG:
1. Persistent ChromaDB Vector Index (local disk, remote HTTP server, or Chroma Cloud)
2. Sparse Keyword Index (BM25 with an English/Hindi/Hinglish multilingual tokenizer)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import docx  # Reads .docx files (python-docx)
import pymupdf  # Fast Unicode-safe PDF reader for English/Hindi/Hinglish (PyMuPDF)
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

load_dotenv()

# Free multilingual embedding model supporting English, Hindi (Devanagari), and Hinglish
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "multilingual_rag"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the free HuggingFace multilingual embedding model on CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def tokenize_multilingual(text: str) -> List[str]:
    r"""
    Tokenize English, Hinglish, and Hindi (Devanagari Unicode \u0900-\u097F) text.
    Standard \w+ splits Hindi words on vowel signs/matras, whereas [\w\u0900-\u097F]+
    keeps full Devanagari words intact as single tokens for accurate BM25 matching.
    """
    return re.findall(r"[\w\u0900-\u097F]+", text.lower())


def get_chroma_client(persist_dir: str | None = None):
    """
    Create a ChromaDB client based on CHROMA_MODE:
    - local (default): PersistentClient storing data permanently in 'chroma_db/'
    - http: HttpClient connecting to a live ChromaDB server (Render / Railway / VPS)
    - cloud: CloudClient connecting to managed Chroma Cloud (trychroma.com)
    """
    load_dotenv(override=True)
    mode = os.getenv("CHROMA_MODE", "local").strip().lower()

    if mode == "http":
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        ssl = os.getenv("CHROMA_SSL", "false").strip().lower() == "true"
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)

    if mode == "cloud":
        return chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )

    target_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
    os.makedirs(target_dir, exist_ok=True)
    return chromadb.PersistentClient(path=target_dir)


def get_chroma_vectorstore(
    embeddings: HuggingFaceEmbeddings | None = None,
    persist_dir: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Open or create the ChromaDB collection wrapped for LangChain."""
    if embeddings is None:
        embeddings = get_embeddings()

    client = get_chroma_client(persist_dir=persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _load_pdf(file_path: str, file_name: str) -> List[Document]:
    """Extract text page by page from a PDF and store 1-based page numbers."""
    documents: List[Document] = []
    with pymupdf.open(file_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_index + 1,
                        },
                    )
                )
    return documents


def _load_docx(file_path: str, file_name: str) -> List[Document]:
    """Extract paragraphs and tables from a DOCX file."""
    doc = docx.Document(file_path)
    parts: List[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)

    full_text = "\n".join(parts).strip()
    if not full_text:
        return []

    return [
        Document(
            page_content=full_text,
            metadata={"source": file_name, "page": 1},
        )
    ]


def _load_txt(file_path: str, file_name: str) -> List[Document]:
    """Read a UTF-8 TXT file safely."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": file_name, "page": 1},
        )
    ]


def _load_json(file_path: str, file_name: str) -> List[Document]:
    """Read a JSON file into readable text while preserving Unicode/Hindi characters."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    formatted_text = json.dumps(data, indent=2, ensure_ascii=False).strip()
    if not formatted_text:
        return []

    return [
        Document(
            page_content=formatted_text,
            metadata={"source": file_name, "page": 1},
        )
    ]


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load PDF, DOCX, TXT, or JSON files into LangChain Document objects."""
    all_docs: List[Document] = []

    for path in file_paths:
        file_name = Path(path).name
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            all_docs.extend(_load_pdf(path, file_name))
        elif ext == ".docx":
            all_docs.extend(_load_docx(path, file_name))
        elif ext == ".txt":
            all_docs.extend(_load_txt(path, file_name))
        elif ext == ".json":
            all_docs.extend(_load_json(path, file_name))
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only PDF, DOCX, TXT, and JSON are supported.")

    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into 1000-character chunks with 200-character overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", ". ", "? ", "! ", " ", ""],
    )
    return text_splitter.split_documents(documents)


def build_bm25_index(chunks: List[Document]) -> Tuple[BM25Okapi | None, List[Document]]:
    """Build a BM25 sparse keyword index over document chunks."""
    if not chunks:
        return None, []
    tokenized_corpus = [tokenize_multilingual(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def build_bm25_from_vectorstore(vectorstore: Chroma) -> Tuple[BM25Okapi | None, List[Document]]:
    """Rebuild the BM25 index from all chunks currently stored in ChromaDB."""
    stored = vectorstore.get(include=["documents", "metadatas"])
    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    chunks: List[Document] = []
    for text, meta in zip(texts, metadatas):
        if text:
            chunks.append(Document(page_content=text, metadata=meta or {}))

    return build_bm25_index(chunks)


def get_chroma_stats(vectorstore: Chroma) -> Dict[str, Any]:
    """Return a summary of all saved filenames and total chunk count in ChromaDB."""
    stored = vectorstore.get(include=["metadatas"])
    metadatas = stored.get("metadatas") or []
    files = sorted({m.get("source", "Unknown") for m in metadatas if m})
    return {
        "files": files,
        "total_chunks": len(metadatas),
    }


def clear_chroma_index(
    embeddings: HuggingFaceEmbeddings | None = None,
    persist_dir: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Delete and recreate the ChromaDB collection."""
    client = get_chroma_client(persist_dir=persist_dir)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return get_chroma_vectorstore(
        embeddings=embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


def build_vector_index(
    file_paths: List[str],
    persist_dir: str | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[Chroma, Tuple[BM25Okapi | None, List[Document]], int, int]:
    """
    Main ingestion pipeline (Persistent ChromaDB + Hybrid BM25):
    1. Load files
    2. Split into chunks
    3. Replace any existing chunks for the same filename to prevent duplicates
    4. Save new chunks permanently into ChromaDB
    5. Refresh the BM25 index across the entire ChromaDB collection
    """
    raw_docs = load_documents(file_paths)
    if not raw_docs:
        raise ValueError(
            "No extractable text was found in the uploaded file(s). "
            "If this is a scanned image PDF, please run OCR first."
        )

    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = get_chroma_vectorstore(
        embeddings=embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    # Remove older chunks from the same filename so re-uploading updates cleanly
    uploaded_sources = {Path(p).name for p in file_paths}
    for source_name in uploaded_sources:
        try:
            vectorstore._collection.delete(where={"source": source_name})
        except Exception:
            pass

    # Add chunks in batches (safe for 400-page PDFs)
    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[start : start + batch_size])

    # Refresh BM25 across all documents in ChromaDB
    bm25_data = build_bm25_from_vectorstore(vectorstore)

    return vectorstore, bm25_data, len(raw_docs), len(chunks)


def load_vector_index(
    persist_dir: str | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[Chroma, Tuple[BM25Okapi | None, List[Document]]]:
    """Load the existing ChromaDB collection and build its matching BM25 index."""
    vectorstore = get_chroma_vectorstore(
        embeddings=embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    bm25_data = build_bm25_from_vectorstore(vectorstore)
    return vectorstore, bm25_data


if __name__ == "__main__":
    import sys

    targets: List[str] = []
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_dir():
                targets.extend(
                    str(f)
                    for f in sorted(p.iterdir())
                    if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".json"}
                )
            else:
                targets.append(str(p))
    else:
        data_dir = Path("data")
        if data_dir.exists():
            targets = [
                str(f)
                for f in sorted(data_dir.iterdir())
                if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt", ".json"}
            ]

    if not targets:
        print("Usage: python ingest.py <file_or_folder> (or place files inside the 'data/' folder)")
        sys.exit(1)

    _, bm25_data, doc_count, chunk_count = build_vector_index(targets)
    print(
        f"Success! Saved {chunk_count} chunks from {len(targets)} file(s) ({doc_count} pages/sections) "
        f"into ChromaDB. Total DB chunks: {len(bm25_data[1])}."
    )
