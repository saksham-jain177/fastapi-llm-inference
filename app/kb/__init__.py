"""
Internal Knowledge Base (KB) evidence source.

Truth-First Principle:
The Orchestrator's `decision == "model"` path requires real evidence. This
module provides an internal, self-hosted evidence store backed by chromadb so
general queries can be grounded against curated documents BEFORE external
search is attempted.

Design:
- Singleton KB (mirrors get_knowledge_gate / get_data_collector pattern).
- Lazy chroma client + PersistentClient at data/chroma_kb (env: KB_PERSIST_DIR).
- Embeddings via the shared sentence-transformers model already used across
  the codebase (all-MiniLM-L6-v2). Falls back to chroma's default embedding
  function if sentence-transformers is unavailable.
- Ingestion: chunk documents on blank lines with a max-char budget.
- Retrieval: similarity search returning (text, metadata, distance) tuples;
  a query only counts as evidence when its best match clears KB_MIN_SIMILARITY
  (cosine-distance-derived similarity threshold).

No network access. No side effects beyond the local persist directory.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from app.observability.logging_setup import get_logger

logger = get_logger("kb")


def _persist_dir() -> Path:
    return Path(os.getenv("KB_PERSIST_DIR", "data/chroma_kb"))


def _collection_name() -> str:
    return os.getenv("KB_COLLECTION", "internal_kb")


def min_similarity() -> float:
    """Minimum similarity (1 - cosine distance) for a match to count as evidence."""
    return float(os.getenv("KB_MIN_SIMILARITY", "0.25"))


class InternalKB:
    """
    Internal document knowledge base for evidence-backed answering.

    Wraps a persistent chromadb collection. All public methods are safe to
    call when chromadb or the embedding model is unavailable — they degrade
    to empty results so callers can fall back to external RAG.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_model = None

    # ---- lazy resources -------------------------------------------------

    def _get_embedding_model(self):
        """Lazily load the shared sentence-transformer model (None if unavailable)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                logger.warning("embedding model unavailable: %s", e)
                self._embedding_model = False  # sentinel: failed load
        return self._embedding_model or None

    def _get_collection(self):
        """Lazily open the chroma collection (None if chromadb unavailable)."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.utils import embedding_functions

                persist = str(_persist_dir())
                Path(persist).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist)

                model = self._get_embedding_model()
                if model is not None:
                    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                else:
                    ef = None  # chroma default ONNX MiniLM

                self._collection = self._client.get_or_create_collection(
                    name=_collection_name(),
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("chromadb unavailable, KB disabled: %s", e)
                self._collection = False  # sentinel: failed init
        return self._collection or None

    # ---- ingestion -------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, max_chunk_chars: int = 1000) -> List[str]:
        """Split a document into chunks on blank lines within a char budget."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chunk_chars and current:
                chunks.append(current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(current)
        return chunks

    def ingest(
        self,
        text: str,
        doc_id: str,
        source: str = "manual",
        upsert: bool = True,
    ) -> int:
        """
        Ingest one document. Returns number of chunks stored; 0 on failure.

        Args:
            text: Raw document text (chunked internally).
            doc_id: Stable unique ID for the document (chunks get suffixes).
            source: Free-form provenance label stored in metadata.
            upsert: Replace existing chunks of doc_id before inserting.
        """
        collection = self._get_collection()
        if collection is None or not text.strip():
            return 0

        if upsert:
            try:
                collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass  # first ingest: nothing to delete

        chunks = self.chunk_text(text)
        ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "source": source, "chunk": i} for i in range(len(chunks))]
        try:
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        except Exception as e:
            logger.error("failed to ingest %r: %s", doc_id, e)
            return 0
        return len(chunks)

    def remove(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        collection = self._get_collection()
        if collection is None:
            return False
        try:
            collection.delete(where={"doc_id": doc_id})
            return True
        except Exception as e:
            logger.error("failed to remove %r: %s", doc_id, e)
            return False

    def count(self) -> int:
        collection = self._get_collection()
        if collection is None:
            return 0
        try:
            return collection.count()
        except Exception:
            return 0

    # ---- retrieval ---------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> Tuple[bool, List[dict]]:
        """
        Search the KB for evidence relevant to the query.

        Returns:
            (has_evidence, results) where results is a list of
            {"text", "doc_id", "source", "similarity"} sorted best-first.
            has_evidence is True only when the best match clears
            KB_MIN_SIMILARITY. Empty/failed KB yields (False, []).
        """
        collection = self._get_collection()
        if collection is None or not query.strip() or collection.count() == 0:
            return False, []

        try:
            res = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
        except Exception as e:
            logger.error("retrieval failed: %s", e)
            return False, []

        results = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            results.append(
                {
                    "text": text,
                    "doc_id": (meta or {}).get("doc_id"),
                    "source": (meta or {}).get("source"),
                    "similarity": 1.0 - float(dist),
                }
            )

        has_evidence = bool(results) and results[0]["similarity"] >= min_similarity()
        return has_evidence, results


# Global instance (mirrors other app-level singletons)
_internal_kb: Optional[InternalKB] = None


def get_internal_kb() -> InternalKB:
    global _internal_kb
    if _internal_kb is None:
        _internal_kb = InternalKB()
    return _internal_kb


def reset_internal_kb():
    global _internal_kb
    _internal_kb = None
