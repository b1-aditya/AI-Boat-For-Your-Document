"""
app.py
------
Streamlit UI supporting automatic indexing on file upload into persistent ChromaDB,
Hybrid Search (ChromaDB MMR + BM25 + RRF), FlashRank Multilingual Reranking,
strict English/Hindi/Hinglish language matching, word-by-word streaming output,
chat history, and an expandable Sources & Page Numbers section.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Patch sqlite3 with pysqlite3 on Linux (Streamlit Cloud) before importing ChromaDB
try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv
from google import genai
from groq import Groq

from ingest import (
    build_bm25_from_vectorstore,
    build_vector_index,
    clear_chroma_index,
    get_chroma_stats,
    get_embeddings,
    load_vector_index,
)
from rag import DEFAULT_GOOGLE_MODEL, DEFAULT_GROQ_MODEL, get_reranker, stream_question

# Load environment variables from .env for local execution
load_dotenv()

st.set_page_config(
    page_title="Hybrid Multilingual RAG (ChromaDB + Reranker)",
    page_icon="📚",
    layout="wide",
)


def get_api_key(provider: str) -> str | None:
    """Read the API key from .env first, falling back to Streamlit Cloud secrets."""
    load_dotenv(override=True)
    key_names = (
        ("GOOGLE_API_KEY", "GEMINI_API_KEY") if provider == "google" else ("GROQ_API_KEY",)
    )

    for key_name in key_names:
        env_key = os.getenv(key_name)
        if env_key:
            return env_key

    try:
        for key_name in key_names:
            if key_name in st.secrets:
                return st.secrets[key_name]
    except Exception:
        pass

    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_available_google_models(api_key: str | None) -> list[str]:
    """Fetch active text-generation Gemini models available for the configured API key."""
    fallback_models = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]
    if not api_key:
        return fallback_models

    try:
        client = genai.Client(api_key=api_key)
        excluded = ("image", "nano-banana", "tts", "audio", "embedding", "aqa", "gemini-2.5")
        chat_models = []
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = m.supported_actions or []
            if (
                "gemini" in name.lower()
                and "generateContent" in actions
                and not any(bad in name.lower() for bad in excluded)
            ):
                chat_models.append(name)

        chat_models = sorted(set(chat_models), reverse=True)
        if not chat_models:
            return fallback_models
        if DEFAULT_GOOGLE_MODEL in chat_models:
            chat_models.remove(DEFAULT_GOOGLE_MODEL)
            chat_models.insert(0, DEFAULT_GOOGLE_MODEL)
        return chat_models
    except Exception:
        return fallback_models


@st.cache_data(ttl=300, show_spinner=False)
def get_available_groq_models(api_key: str | None) -> list[str]:
    """Fetch active chat LLM models available on the configured Groq account."""
    fallback_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    if not api_key:
        return fallback_models

    try:
        models = Groq(api_key=api_key).models.list().data
        excluded_keywords = ("whisper", "orpheus", "guard", "safeguard", "tts")
        chat_models = sorted(
            m.id for m in models if not any(bad in m.id.lower() for bad in excluded_keywords)
        )
        if not chat_models:
            return fallback_models
        if DEFAULT_GROQ_MODEL in chat_models:
            chat_models.remove(DEFAULT_GROQ_MODEL)
            chat_models.insert(0, DEFAULT_GROQ_MODEL)
        return chat_models
    except Exception:
        return fallback_models


@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def load_cached_embeddings():
    """Load and cache the HuggingFace embedding model once in memory."""
    return get_embeddings()


@st.cache_resource(show_spinner="Loading multilingual reranker model...")
def load_cached_reranker():
    """Load and cache the FlashRank cross-encoder reranker once in memory."""
    return get_reranker()


@st.cache_resource(show_spinner="Connecting to ChromaDB index...")
def load_cached_vectorstore():
    """Load and cache the ChromaDB vectorstore once per Streamlit server process."""
    embeddings = load_cached_embeddings()
    vectorstore, _ = load_vector_index(embeddings=embeddings)
    return vectorstore


def init_session_state():
    """Initialize chat history and load any existing ChromaDB + BM25 index automatically."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_file_signatures" not in st.session_state:
        st.session_state.indexed_file_signatures = set()
    if "last_ingest_notice" not in st.session_state:
        st.session_state.last_ingest_notice = None
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        vectorstore = load_cached_vectorstore()
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25_data = build_bm25_from_vectorstore(vectorstore)


def auto_ingest_uploaded_files(uploaded_files) -> None:
    """
    Automatically detect newly uploaded files, chunk and embed them into ChromaDB,
    and refresh the BM25 index without requiring a manual button click.
    """
    if not uploaded_files:
        return

    new_files = [
        f for f in uploaded_files if (f.name, f.size) not in st.session_state.indexed_file_signatures
    ]
    if not new_files:
        return

    start_time = time.time()
    embeddings = load_cached_embeddings()
    load_cached_reranker()

    with st.spinner("Indexing uploaded file(s) into ChromaDB and BM25..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_paths = []
            for uploaded_file in new_files:
                file_path = os.path.join(tmp_dir, Path(uploaded_file.name).name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

            vectorstore, bm25_data, num_docs, num_chunks = build_vector_index(
                file_paths=saved_paths,
                embeddings=embeddings,
                chunk_size=1000,
                chunk_overlap=200,
            )

        elapsed = round(time.time() - start_time, 1)
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25_data = bm25_data
        for f in new_files:
            st.session_state.indexed_file_signatures.add((f.name, f.size))

        st.session_state.last_ingest_notice = (
            f"Hybrid Index Ready! Saved {len(new_files)} file(s) "
            f"({num_docs} pages/sections -> {num_chunks} chunks) to ChromaDB in {elapsed}s."
        )


def render_sources_expander(sources: list[dict]):
    """Render an expandable Sources, Page Numbers, and Rerank Scores panel below an answer."""
    if not sources:
        return

    with st.expander("View Sources, Page Numbers & Rerank Scores"):
        for i, src in enumerate(sources, start=1):
            score_text = f" | **Rerank Score:** `{src['score']}`" if src.get("score") is not None else ""
            st.markdown(
                f"**{i}. File:** `{src['source']}` | **Page:** `{src['page']}`{score_text}"
            )
            st.caption(src["snippet"])
            if i < len(sources):
                st.divider()


def main():
    init_session_state()

    st.title("📚 Hybrid Multilingual Document Q&A (ChromaDB + Reranker)")
    st.caption(
        "ChromaDB MMR + BM25 Hybrid Search + FlashRank Multilingual Reranker + Google Gemini LLM "
        "(English, Hindi & Hinglish)"
    )

    # ---------------- SIDEBAR: Auto-Upload & Settings ----------------
    with st.sidebar:
        st.header("1. Upload Documents (Auto-Index)")

        uploaded_files = st.file_uploader(
            "Browse and upload files (Hybrid Index builds automatically)",
            type=["pdf", "docx", "txt", "json"],
            accept_multiple_files=True,
            help="Upload PDF (up to 400 pages), DOCX, TXT, or JSON files. Saved permanently to ChromaDB.",
        )

        try:
            auto_ingest_uploaded_files(uploaded_files)
        except Exception as e:
            st.error(f"Auto-indexing failed: {e}")

        if st.session_state.last_ingest_notice:
            st.success(st.session_state.last_ingest_notice)

        st.header("2. LLM Settings")
        provider_label = st.selectbox(
            "LLM Provider",
            options=["Google Gemini", "Groq"],
            index=0,
        )
        provider = "google" if provider_label == "Google Gemini" else "groq"
        api_key = get_api_key(provider)

        if provider == "google":
            available_models = get_available_google_models(api_key)
        else:
            available_models = get_available_groq_models(api_key)

        model_choice = st.selectbox(
            "Model",
            options=available_models,
            index=0,
            help="Active chat models available on your API key.",
        )

        stats = get_chroma_stats(st.session_state.vectorstore)
        if stats["total_chunks"] > 0:
            st.info(
                f"**Status:** Hybrid Index Ready (ChromaDB)\n\n"
                f"**Saved Files ({len(stats['files'])}):** {', '.join(stats['files'])}\n\n"
                f"**Total Chunks:** {stats['total_chunks']}\n\n"
                f"**Pipeline:** Chroma MMR + BM25 -> RRF -> FlashRank Top-5"
            )
        else:
            st.info("ChromaDB is empty. Upload any file above to get started.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Reset ChromaDB", use_container_width=True):
                embeddings = load_cached_embeddings()
                st.session_state.vectorstore = clear_chroma_index(embeddings=embeddings)
                load_cached_vectorstore.clear()
                st.session_state.bm25_data = (None, [])
                st.session_state.indexed_file_signatures = set()
                st.session_state.last_ingest_notice = None
                st.session_state.messages = []
                st.rerun()

        if not api_key:
            missing_var = "GOOGLE_API_KEY" if provider == "google" else "GROQ_API_KEY"
            st.warning(
                f"{missing_var} is missing! Add it to your `.env` file locally or in "
                "`App Settings -> Secrets` on Streamlit Cloud."
            )

    # ---------------- MAIN CHAT AREA ----------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources_expander(msg["sources"])

    user_question = st.chat_input("Ask a question in English, Hindi, or Hinglish...")

    if user_question:
        if not api_key:
            missing_var = "GOOGLE_API_KEY" if provider == "google" else "GROQ_API_KEY"
            st.error(f"{missing_var} is not configured. Please add your API key first.")
            return

        if not st.session_state.bm25_data or not st.session_state.bm25_data[1]:
            st.warning("ChromaDB has no indexed documents yet. Please upload a file in the sidebar.")
            return

        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            try:
                reranker = load_cached_reranker()
                clean_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                with st.spinner("Searching documents and reranking top results..."):
                    word_stream, reranked_docs = stream_question(
                        vectorstore=st.session_state.vectorstore,
                        question=user_question,
                        bm25_data=st.session_state.bm25_data,
                        reranker=reranker,
                        chat_history=clean_history,
                        provider=provider,
                        api_key=api_key,
                        model_name=model_choice,
                    )

                # Stream the generated answer word-by-word directly into the chat bubble
                answer = st.write_stream(word_stream)
                if not isinstance(answer, str):
                    answer = str(answer)
                answer = answer.strip()

                formatted_sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "score": doc.metadata.get("rerank_score"),
                        "snippet": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
                    }
                    for doc in reranked_docs
                ]

                if formatted_sources:
                    render_sources_expander(formatted_sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": formatted_sources,
                    }
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
