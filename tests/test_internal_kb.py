"""Tests for the internal knowledge base (chromadb-backed evidence source)."""

import os

os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"
os.environ.setdefault("KB_PERSIST_DIR", "data/test_chroma_kb")

import pytest

from app.kb import InternalKB, get_internal_kb, reset_internal_kb
from app.kb.seed_kb import SAMPLE_CORPUS, seed


@pytest.fixture()
def kb(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_PERSIST_DIR", str(tmp_path / "kb"))
    reset_internal_kb()
    instance = InternalKB()
    yield instance
    reset_internal_kb()


class TestChunking:
    def test_chunks_on_blank_lines_within_budget(self):
        text = "\n\n".join("para %d %s" % (i, "x" * 200) for i in range(10))
        chunks = InternalKB.chunk_text(text, max_chunk_chars=500)
        assert len(chunks) >= 2
        assert all(len(c) <= 600 for c in chunks)  # budget + one-paragraph slack

    def test_single_paragraph_stays_intact(self):
        chunks = InternalKB.chunk_text("one single paragraph")
        assert chunks == ["one single paragraph"]

    def test_empty_text_yields_no_chunks(self):
        assert InternalKB.chunk_text("  \n\n  ") == []


class TestIngestAndRetrieve:
    def test_ingest_returns_chunk_count(self, kb):
        n = kb.ingest(SAMPLE_CORPUS["company_faq"], doc_id="faq", source="test")
        assert n >= 1
        assert kb.count() == n

    def test_upsert_replaces_old_chunks(self, kb):
        kb.ingest("old content here", doc_id="d1")
        kb.ingest("entirely new content now", doc_id="d1")
        assert kb.count() == 1
        has_evidence, results = kb.retrieve("entirely new content")
        assert has_evidence and results[0]["text"] == "entirely new content now"

    def test_remove_document(self, kb):
        kb.ingest("some text", doc_id="gone")
        assert kb.remove("gone") is True
        assert kb.count() == 0

    def test_relevant_query_clears_evidence(self, kb):
        seed(kb)
        has_evidence, results = kb.retrieve("How many vacation days do Acme employees get?")
        assert has_evidence is True
        assert results[0]["doc_id"] == "company_faq"
        assert results[0]["similarity"] > 0.25
        assert set(results[0].keys()) == {"text", "doc_id", "source", "similarity"}

    def test_irrelevant_query_denies_evidence(self, kb):
        seed(kb)
        has_evidence, _ = kb.retrieve("quantum chromodynamics renormalization group equations")
        assert has_evidence is False

    def test_empty_kb_has_no_evidence(self, kb):
        has_evidence, results = kb.retrieve("anything at all")
        assert (has_evidence, results) == (False, [])

    def test_blank_query_has_no_evidence(self, kb):
        seed(kb)
        assert kb.retrieve("   ")[0] is False


class TestOrchestratorIntegration:
    async def test_model_path_uses_kb_evidence(self, tmp_path, monkeypatch):
        """KB evidence flips decision=='model' path on with real grounding."""
        from unittest.mock import AsyncMock, MagicMock, patch

        monkeypatch.setenv("KB_PERSIST_DIR", str(tmp_path / "kb"))
        reset_internal_kb()
        seed()

        from app.routing.orchestrator import Orchestrator

        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_reasoner"), \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector:
            mock_router = MagicMock()
            mock_router.classify.return_value = ("general", 0.9)  # high semantic confidence
            mock_get_router.return_value = mock_router
            mock_search.return_value = ("Context", [])  # must NOT be called
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector

            orch = Orchestrator()
            result = await orch.route_and_execute(
                "What support tiers does Acme Corp offer?"
            )

            assert result["mode"] == "model"
            assert result["evidence"] == "internal_kb"
            assert result["kb_sources"][0]["doc_id"] in SAMPLE_CORPUS
            mock_search.assert_not_called()  # internal KB answered before external RAG

        reset_internal_kb()

    async def test_no_kb_match_falls_back_to_rag(self, tmp_path, monkeypatch):
        """Without a KB match the pipeline degrades to external search as before."""
        from unittest.mock import AsyncMock, MagicMock, patch

        monkeypatch.setenv("KB_PERSIST_DIR", str(tmp_path / "kb"))
        reset_internal_kb()  # empty KB: nothing ingested

        from app.routing.orchestrator import Orchestrator

        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            mock_router = MagicMock()
            mock_router.classify.return_value = ("general", 0.9)
            mock_get_router.return_value = mock_router
            mock_search.return_value = ("Context", [])
            mock_reasoner = AsyncMock()
            mock_reasoner.synthesize_with_context.return_value = "A complete answer."
            mock_get_reasoner.return_value = mock_reasoner
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector

            orch = Orchestrator()
            result = await orch.route_and_execute("What support tiers does Acme offer?")

            assert result["mode"] == "rag-external"
            mock_search.assert_called_once()

        reset_internal_kb()


class TestSingleton:
    def test_get_internal_kb_is_singleton(self):
        reset_internal_kb()
        assert get_internal_kb() is get_internal_kb()
        reset_internal_kb()
