r"""
api.py
------
FastAPI REST API server for the Multilingual Hybrid RAG system,
protected with your custom `RAG_API_KEY` via the `X-API-Key` header.

Run locally:
    .\.venv\Scripts\uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Swagger Docs:
    http://localhost:8000/docs
"""

import os
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ingest import build_vector_index, get_chroma_stats, get_embeddings, load_vector_index
from rag import ask_question, get_reranker

load_dotenv()

app = FastAPI(
    title="Multilingual Hybrid RAG API (ChromaDB + Reranker)",
    description="English, Hindi & Hinglish Hybrid RAG API secured with X-API-Key.",
    version="1.0.0",
)

# Cached singleton instances loaded on startup
_embeddings = None
_reranker = None
_vectorstore = None
_bm25_data = None


def _ensure_initialized():
    global _embeddings, _reranker, _vectorstore, _bm25_data
    if _embeddings is None:
        _embeddings = get_embeddings()
    if _reranker is None:
        _reranker = get_reranker()
    if _vectorstore is None:
        _vectorstore, _bm25_data = load_vector_index(embeddings=_embeddings)


def verify_rag_api_key(x_api_key: str = Header(..., description="Your RAG_API_KEY from .env")):
    """Validate the X-API-Key header against RAG_API_KEY in .env."""
    load_dotenv(override=True)
    expected_key = os.getenv("RAG_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG_API_KEY is not configured on the server (.env).",
        )
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid RAG API Key.",
        )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question in English, Hindi, or Hinglish")
    provider: str = Field(default="google", description="'google' or 'groq'")
    model: str | None = Field(default=None, description="Optional model override")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class SourceItem(BaseModel):
    source: str
    page: int | str
    rerank_score: float | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.on_event("startup")
def startup_event():
    _ensure_initialized()


@app.get("/health")
def health_check():
    """Public health check endpoint."""
    _ensure_initialized()
    stats = get_chroma_stats(_vectorstore)
    return {
        "status": "ok",
        "vectordb": "ChromaDB",
        "saved_files": stats["files"],
        "total_chunks": stats["total_chunks"],
    }


@app.get("/documents", dependencies=[Depends(verify_rag_api_key)])
def list_documents():
    """List all permanently saved filenames and chunk counts in ChromaDB."""
    _ensure_initialized()
    return get_chroma_stats(_vectorstore)


@app.post("/ingest", dependencies=[Depends(verify_rag_api_key)])
async def ingest_files(files: List[UploadFile] = File(...)):
    """Upload and index documents into ChromaDB and refresh the BM25 index."""
    global _vectorstore, _bm25_data
    _ensure_initialized()

    allowed_exts = {".pdf", ".docx", ".txt", ".json"}
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        for upload in files:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in allowed_exts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file '{upload.filename}'. Allowed: PDF, DOCX, TXT, JSON",
                )
            file_path = os.path.join(tmp_dir, Path(upload.filename).name)
            content = await upload.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_paths.append(file_path)

        _vectorstore, _bm25_data, num_docs, num_chunks = build_vector_index(
            file_paths=saved_paths,
            embeddings=_embeddings,
            chunk_size=1000,
            chunk_overlap=200,
        )

    stats = get_chroma_stats(_vectorstore)
    return {
        "status": "Hybrid Index Ready",
        "processed_pages_or_sections": num_docs,
        "new_chunks_added": num_chunks,
        "total_db_chunks": stats["total_chunks"],
        "saved_files": stats["files"],
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_rag_api_key)])
def ask_rag(payload: AskRequest):
    """Protected Q&A endpoint using ChromaDB MMR + BM25 + FlashRank Reranker + Gemini/Groq."""
    _ensure_initialized()
    if not _bm25_data or not _bm25_data[1]:
        raise HTTPException(
            status_code=400,
            detail="ChromaDB is empty. Upload documents via /ingest first.",
        )

    answer, reranked_docs = ask_question(
        vectorstore=_vectorstore,
        question=payload.question,
        bm25_data=_bm25_data,
        reranker=_reranker,
        chat_history=payload.chat_history,
        provider=payload.provider,
        model_name=payload.model,
    )

    sources = [
        SourceItem(
            source=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page", "N/A"),
            rerank_score=doc.metadata.get("rerank_score"),
            snippet=doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
        )
        for doc in reranked_docs
    ]
    return AskResponse(answer=answer, sources=sources)
