"""
rag.py
------
Handles Hybrid Retrieval (ChromaDB MMR + Sparse BM25 + Reciprocal Rank Fusion),
Multilingual Cross-Encoder Reranking (FlashRank), strict language matching
(English -> English, Hindi -> Hindi, Hinglish -> Hinglish), and word-by-word
streaming or standard generation via Google Gemini / Groq LLM.
"""

import os
import re
from typing import Dict, Iterator, List, Tuple

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

from ingest import build_bm25_from_vectorstore, tokenize_multilingual

load_dotenv()

NOT_FOUND_MESSAGE = "Is document mein ye information nahi hai"

DEFAULT_GOOGLE_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GOOGLE_FALLBACK_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)

RERANKER_MODEL_NAME = "ms-marco-MultiBERT-L-12"

# Common Romanized Hindi words used to distinguish Hinglish from pure English
HINGLISH_MARKERS = {
    "kya", "kaise", "kaun", "kaunse", "kaunsi", "kab", "kahan", "kyun", "kyu", "kyunki",
    "hai", "hain", "ho", "tha", "the", "thi", "mein", "me", "ko", "ka", "ki", "ke",
    "aur", "ya", "nahi", "nhi", "batao", "btaiye", "bataiye", "samjhao", "samjha",
    "likha", "diya", "banao", "karo", "kr", "kare", "karen", "iska", "uska", "ye",
    "wo", "vo", "kitna", "kitne", "chahiye", "sakte", "sakta", "sakti", "hoga", "hogi",
    "mera", "meri", "apna", "apni", "mujhe", "muje", "tum", "aap", "yahan", "wahan",
    "koi", "sab", "baare", "bare", "bhi", "bhe", "toh", "wala", "wali", "wale",
    "dikhao", "bolo", "hinglish", "karke", "krke", "raha", "rahi", "rahe", "hota",
}


def detect_question_language(question: str) -> str:
    """
    Detect whether the user asked the question in Hindi (Devanagari),
    Hinglish (Romanized Hindi + English), or pure English.
    """
    if re.search(r"[\u0900-\u097F]", question):
        return "Hindi"

    words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    if words & HINGLISH_MARKERS:
        return "Hinglish"

    return "English"


def _get_language_instruction(language: str) -> str:
    """Return strict response language instructions for the LLM."""
    if language == "Hindi":
        return (
            "The user asked in Hindi. Respond strictly in clear Hindi (Devanagari script only), "
            "even if the retrieved document text is in English or Hinglish."
        )
    if language == "Hinglish":
        return (
            "The user asked in Hinglish. Respond strictly in Hinglish (Hindi written in Roman/English "
            "script mixed naturally with English terms), even if the retrieved document text is in "
            "pure English or Devanagari Hindi."
        )
    return (
        "The user asked in English. Respond strictly in clear, natural English only, "
        "even if the retrieved document text is in Hindi or Hinglish."
    )


def get_reranker() -> Ranker:
    """Load the lightweight (~150MB ONNX) FlashRank multilingual cross-encoder reranker."""
    return Ranker(model_name=RERANKER_MODEL_NAME)


def get_llm(
    provider: str = "google",
    api_key: str | None = None,
    model_name: str | None = None,
) -> Runnable:
    """
    Initialize the LLM for the selected provider ('google' or 'groq').
    Includes automatic fallback models for Google Gemini to handle transient 503 load spikes.
    """
    load_dotenv(override=True)

    if provider == "google":
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please add it to your .env file or Streamlit Secrets."
            )
        selected_model = model_name or DEFAULT_GOOGLE_MODEL
        primary_llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0,
            api_key=resolved_key,
            max_retries=1,
        )
        fallback_llms = [
            ChatGoogleGenerativeAI(
                model=m,
                temperature=0,
                api_key=resolved_key,
                max_retries=1,
            )
            for m in GOOGLE_FALLBACK_MODELS
            if m != selected_model
        ]
        return primary_llm.with_fallbacks(fallback_llms)

    if provider == "groq":
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please add it to your .env file or Streamlit Secrets."
            )
        return ChatGroq(
            model=model_name or DEFAULT_GROQ_MODEL,
            temperature=0,
            api_key=resolved_key,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _bm25_search(
    bm25_data: Tuple[BM25Okapi | None, List[Document]],
    query: str,
    k: int = 10,
) -> List[Document]:
    """Retrieve top-k exact keyword-matching chunks from the BM25 index."""
    bm25, corpus_docs = bm25_data
    if bm25 is None or not corpus_docs:
        return []

    query_tokens = tokenize_multilingual(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [corpus_docs[i] for i in ranked_indices[:k] if scores[i] > 0]


def _hybrid_retrieve_with_rrf(
    vectorstore: Chroma,
    bm25_data: Tuple[BM25Okapi | None, List[Document]],
    query: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    rrf_top_n: int = 15,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Hybrid Retrieval + Reciprocal Rank Fusion (RRF):
    - Dense Retriever (ChromaDB MMR): semantic similarity + diversity
    - Sparse Retriever (BM25): exact keywords, names, numbers, and Hindi/English terms
    - Clamps fetch_k and k to total_docs so small files do not trigger HNSW errors
    """
    total_docs = len(bm25_data[1])
    if total_docs == 0:
        return []

    actual_fetch_k = min(30, total_docs)
    actual_dense_k = min(dense_k, actual_fetch_k)

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": actual_dense_k,
            "fetch_k": actual_fetch_k,
            "lambda_mult": 0.5,
        },
    )
    dense_docs = mmr_retriever.invoke(query)
    sparse_docs = _bm25_search(bm25_data, query, k=min(sparse_k, total_docs))

    fused_scores: Dict[tuple, float] = {}
    doc_map: Dict[tuple, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
        doc_map[key] = doc
        fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    for rank, doc in enumerate(sparse_docs, start=1):
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
        doc_map[key] = doc
        fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:rrf_top_n]]


def _rerank_documents(
    query: str,
    candidate_docs: List[Document],
    reranker: Ranker,
    top_k: int = 5,
) -> List[Document]:
    """Score candidate chunks with FlashRank Cross-Encoder and keep the top 5."""
    if not candidate_docs:
        return []

    passages = [
        {"id": idx, "text": doc.page_content}
        for idx, doc in enumerate(candidate_docs)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_results = reranker.rerank(rerank_request)

    top_docs: List[Document] = []
    for item in reranked_results[:top_k]:
        orig_doc = candidate_docs[item["id"]]
        updated_meta = dict(orig_doc.metadata)
        updated_meta["rerank_score"] = round(float(item["score"]), 4)
        top_docs.append(
            Document(
                page_content=orig_doc.page_content,
                metadata=updated_meta,
            )
        )

    return top_docs


def _rewrite_followup_query(
    question: str,
    chat_history: List[Dict[str, str]],
    llm: Runnable,
) -> str:
    """Rewrite a follow-up question into a standalone search query using recent history."""
    if not chat_history:
        return question

    recent_history = chat_history[-4:]
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history
    )

    condense_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, rewrite the latest question "
                "into a standalone search query in the same language. Do NOT answer the question. "
                "If it is already standalone, return it unchanged.",
            ),
            ("human", "Chat History:\n{history}\n\nLatest Question: {question}"),
        ]
    )

    chain = condense_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"history": history_text, "question": question}).strip()
    return rewritten or question


def _format_context(docs: List[Document]) -> str:
    """Format each reranked chunk with its source filename, page number, and relevance score."""
    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score", "N/A")
        formatted_chunks.append(
            f"[Chunk {idx} | Source: {source} | Page: {page} | Relevance: {score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def _prepare_rag_pipeline(
    vectorstore: Chroma,
    question: str,
    bm25_data: Tuple[BM25Okapi | None, List[Document]] | None = None,
    reranker: Ranker | None = None,
    chat_history: List[Dict[str, str]] | None = None,
    provider: str = "google",
    api_key: str | None = None,
    model_name: str | None = None,
) -> Tuple[Runnable | None, Dict[str, str] | None, List[Document]]:
    """
    Shared retrieval, reranking, and prompt preparation step used by both
    ask_question() and stream_question().
    """
    chat_history = chat_history or []
    llm = get_llm(provider=provider, api_key=api_key, model_name=model_name)

    if bm25_data is None:
        bm25_data = build_bm25_from_vectorstore(vectorstore)

    if not bm25_data[1]:
        return None, None, []

    if reranker is None:
        reranker = get_reranker()

    # Step 1: Build standalone search query if chat history exists
    search_query = _rewrite_followup_query(question, chat_history, llm)

    # Step 2: Hybrid Search (ChromaDB MMR + BM25 -> RRF Top 15)
    candidate_docs = _hybrid_retrieve_with_rrf(
        vectorstore=vectorstore,
        bm25_data=bm25_data,
        query=search_query,
        dense_k=10,
        sparse_k=10,
        rrf_top_n=15,
    )
    if not candidate_docs:
        return None, None, []

    # Step 3: Cross-Encoder Reranking -> Top 5 best chunks
    reranked_docs = _rerank_documents(
        query=search_query,
        candidate_docs=candidate_docs,
        reranker=reranker,
        top_k=5,
    )
    if not reranked_docs:
        return None, None, []

    context_text = _format_context(reranked_docs)
    detected_language = detect_question_language(question)
    language_instruction = _get_language_instruction(detected_language)

    # Step 4: Strict grounded prompt with exact language matching and page citations
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict document Q&A assistant supporting English, Hindi, and Hinglish.\n"
                "STRICT RULES:\n"
                "1. Answer ONLY using the facts inside <context>. Never use outside knowledge.\n"
                "2. If the answer is NOT in <context> or there is not enough information, "
                f'reply ONLY with this exact sentence: "{NOT_FOUND_MESSAGE}"\n'
                "3. Always cite the source file name and page number from <context> in every answer, "
                "for example: (Source: filename.pdf, Page: 1).\n"
                "4. Language Rule: {language_instruction}",
            ),
            (
                "human",
                "<context>\n{context}\n</context>\n\nQuestion: {question}",
            ),
        ]
    )

    rag_chain = rag_prompt | llm | StrOutputParser()
    chain_input = {
        "context": context_text,
        "question": question,
        "language_instruction": language_instruction,
    }
    return rag_chain, chain_input, reranked_docs


def ask_question(
    vectorstore: Chroma,
    question: str,
    bm25_data: Tuple[BM25Okapi | None, List[Document]] | None = None,
    reranker: Ranker | None = None,
    chat_history: List[Dict[str, str]] | None = None,
    provider: str = "google",
    api_key: str | None = None,
    model_name: str | None = None,
) -> Tuple[str, List[Document]]:
    """Run the full Hybrid RAG + Reranking pipeline and return the complete answer."""
    rag_chain, chain_input, reranked_docs = _prepare_rag_pipeline(
        vectorstore=vectorstore,
        question=question,
        bm25_data=bm25_data,
        reranker=reranker,
        chat_history=chat_history,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
    )

    if rag_chain is None or chain_input is None:
        return NOT_FOUND_MESSAGE, []

    answer = rag_chain.invoke(chain_input).strip()
    return answer, reranked_docs


def stream_question(
    vectorstore: Chroma,
    question: str,
    bm25_data: Tuple[BM25Okapi | None, List[Document]] | None = None,
    reranker: Ranker | None = None,
    chat_history: List[Dict[str, str]] | None = None,
    provider: str = "google",
    api_key: str | None = None,
    model_name: str | None = None,
) -> Tuple[Iterator[str], List[Document]]:
    """
    Run Hybrid RAG + Reranking and return a word-by-word text stream generator
    along with the top-5 reranked source documents for the UI expander.
    """
    rag_chain, chain_input, reranked_docs = _prepare_rag_pipeline(
        vectorstore=vectorstore,
        question=question,
        bm25_data=bm25_data,
        reranker=reranker,
        chat_history=chat_history,
        provider=provider,
        api_key=api_key,
        model_name=model_name,
    )

    def _word_generator() -> Iterator[str]:
        if rag_chain is None or chain_input is None:
            for word in re.findall(r"\S+\s*", NOT_FOUND_MESSAGE):
                yield word
            return

        # Stream tokens from the LLM and split any multi-word chunk into word-by-word pieces
        for chunk in rag_chain.stream(chain_input):
            if not chunk:
                continue
            parts = re.findall(r"\S+\s*|\s+", chunk)
            for part in parts:
                yield part

    return _word_generator(), reranked_docs
