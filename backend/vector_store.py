"""
vector_store.py — ChromaDB integration for the research assistant.

Responsibilities:
  1. store_research()   — persist a completed report + its source chunks
  2. retrieve_similar() — semantic search over past research
  3. store_chunks()     — embed and store individual source summaries
  4. get_collection_stats() — UI metadata (doc count, topics)

Collections:
  research_reports  — one document per completed report (full markdown)
  source_chunks     — individual source summaries, linked to their report
"""

import os
import hashlib
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Paths ────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ── Embedding model (runs locally, no API key needed) ────────────────────────
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"   # Fast, 384-dim, ~80MB download
        )
    return _embeddings


# ── Raw ChromaDB client (for metadata queries) ────────────────────────────────
_chroma_client = None

def get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


# ── LangChain Chroma wrappers (for semantic search) ──────────────────────────
def get_reports_store() -> Chroma:
    return Chroma(
        client=get_client(),
        collection_name="research_reports",
        embedding_function=get_embeddings(),
    )

def get_chunks_store() -> Chroma:
    return Chroma(
        client=get_client(),
        collection_name="source_chunks",
        embedding_function=get_embeddings(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def store_research(query: str, report: str, summaries: list[str]) -> str:
    """
    Persist a completed research session to ChromaDB.
    Returns the report_id (used to link source_chunks).
    """
    report_id = hashlib.md5(f"{query}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
    timestamp = datetime.utcnow().isoformat()

    # Store the full report
    store = get_reports_store()
    store.add_texts(
        texts=[report],
        metadatas=[{
            "query": query,
            "report_id": report_id,
            "timestamp": timestamp,
            "summary_count": len(summaries),
        }],
        ids=[report_id],
    )

    # Store individual source chunks linked to this report
    if summaries:
        store_chunks(report_id=report_id, query=query, summaries=summaries, timestamp=timestamp)

    return report_id


def store_chunks(report_id: str, query: str, summaries: list[str], timestamp: str):
    """Embed and store individual source summaries."""
    store = get_chunks_store()
    chunk_ids = [f"{report_id}_chunk_{i}" for i in range(len(summaries))]
    store.add_texts(
        texts=summaries,
        metadatas=[{
            "report_id": report_id,
            "query": query,
            "chunk_index": i,
            "timestamp": timestamp,
        } for i in range(len(summaries))],
        ids=chunk_ids,
    )


def retrieve_similar(query: str, k: int = 3) -> list[dict]:
    """
    Semantic search over past research reports.
    Returns a list of {query, snippet, report_id, timestamp, score}.
    """
    store = get_reports_store()
    try:
        results = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return []

    output = []
    for doc, score in results:
        output.append({
            "past_query": doc.metadata.get("query", ""),
            "snippet": doc.page_content[:400] + "…",
            "report_id": doc.metadata.get("report_id", ""),
            "timestamp": doc.metadata.get("timestamp", ""),
            "score": round(float(score), 3),
        })
    return output


def retrieve_relevant_chunks(query: str, k: int = 4) -> list[dict]:
    """
    Semantic search over individual source chunks.
    Used by Writer agent to pull in relevant prior knowledge.
    """
    store = get_chunks_store()
    try:
        results = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        return []

    output = []
    for doc, score in results:
        if score > 0.3:   # Only include reasonably relevant chunks
            output.append({
                "text": doc.page_content,
                "source_query": doc.metadata.get("query", ""),
                "score": round(float(score), 3),
            })
    return output


def get_collection_stats() -> dict:
    """Return metadata for the UI sidebar."""
    client = get_client()
    try:
        reports_col = client.get_or_create_collection("research_reports")
        chunks_col  = client.get_or_create_collection("source_chunks")
        report_count = reports_col.count()
        chunk_count  = chunks_col.count()

        # Get the 5 most recent query topics
        recent_topics = []
        if report_count > 0:
            all_meta = reports_col.get(include=["metadatas"])
            metas = all_meta.get("metadatas", [])
            metas_sorted = sorted(metas, key=lambda m: m.get("timestamp",""), reverse=True)
            recent_topics = [m.get("query","") for m in metas_sorted[:5]]

        return {
            "report_count": report_count,
            "chunk_count": chunk_count,
            "recent_topics": recent_topics,
        }
    except Exception:
        return {"report_count": 0, "chunk_count": 0, "recent_topics": []}
