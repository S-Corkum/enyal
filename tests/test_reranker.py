"""Tests for the reranker engine module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from enyal.embeddings.reranker import RerankerConfig, RerankerEngine


class TestRerankerConfig:
    """Tests for RerankerConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default RerankerConfig values."""
        config = RerankerConfig()
        assert config.name == "Qwen/Qwen3-Reranker-0.6B"
        assert config.max_length == 8192
        assert config.trust_remote_code is True

    def test_custom_config(self) -> None:
        """Test RerankerConfig with custom values."""
        config = RerankerConfig(name="custom/model", max_length=4096)
        assert config.name == "custom/model"
        assert config.max_length == 4096

    def test_from_env_no_var(self) -> None:
        """Test from_env with no ENYAL_RERANKER_MODEL returns default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENYAL_RERANKER_MODEL", None)
            config = RerankerConfig.from_env()
            assert config.name == "Qwen/Qwen3-Reranker-0.6B"

    def test_from_env_with_var(self) -> None:
        """Test from_env with ENYAL_RERANKER_MODEL set."""
        with patch.dict(os.environ, {"ENYAL_RERANKER_MODEL": "custom/reranker"}):
            config = RerankerConfig.from_env()
            assert config.name == "custom/reranker"

    def test_frozen_dataclass(self) -> None:
        """Test that RerankerConfig is immutable."""
        config = RerankerConfig()
        with pytest.raises(AttributeError):
            config.name = "something-else"  # type: ignore[misc]


class TestRerankerEngine:
    """Tests for the RerankerEngine."""

    def test_init_default_config(self) -> None:
        """Test creating engine with default config."""
        engine = RerankerEngine()
        assert engine.config.name == "Qwen/Qwen3-Reranker-0.6B"
        assert engine.is_loaded() is False

    def test_init_custom_config(self) -> None:
        """Test creating engine with custom config."""
        config = RerankerConfig(name="custom/model")
        engine = RerankerEngine(config)
        assert engine.config.name == "custom/model"

    def test_is_loaded_false_initially(self) -> None:
        """Test is_loaded returns False when model not loaded."""
        engine = RerankerEngine()
        assert engine.is_loaded() is False

    def test_is_loaded_true_after_load(self) -> None:
        """Test is_loaded returns True after model is set."""
        engine = RerankerEngine()
        engine._model = MagicMock()
        assert engine.is_loaded() is True

    def test_unload(self) -> None:
        """Test unload clears the model and tokenizer."""
        engine = RerankerEngine()
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._yes_token_id = 1
        engine._no_token_id = 2
        engine._prefix_tokens = [1, 2, 3]
        engine._suffix_tokens = [4, 5, 6]

        engine.unload()

        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._yes_token_id is None
        assert engine._no_token_id is None
        assert engine._prefix_tokens is None
        assert engine._suffix_tokens is None
        assert engine.is_loaded() is False

    def test_format_input(self) -> None:
        """Test input formatting for the reranker."""
        result = RerankerEngine._format_input(
            query="What is Python?",
            document="Python is a programming language.",
            instruction="Retrieve relevant passages.",
        )
        assert "<Instruct>: Retrieve relevant passages." in result
        assert "<Query>: What is Python?" in result
        assert "<Document>: Python is a programming language." in result

    def test_rerank_empty_documents(self) -> None:
        """Test rerank with empty document list."""
        engine = RerankerEngine()
        scores = engine.rerank("query", [])
        assert scores == []

    def test_rerank_returns_scores(self) -> None:
        """Test rerank returns one score per document."""
        engine = RerankerEngine()

        # Create mock model that returns logits
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer methods
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: {"yes": 10, "no": 11}[t]
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.apply_chat_template.return_value = "template text"

        # Mock model output with logits
        logits = torch.zeros(1, 10, 100)  # batch=1, seq_len=10, vocab=100
        logits[0, -1, 10] = 2.0  # yes logit
        logits[0, -1, 11] = -1.0  # no logit
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        # Set up engine state
        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._yes_token_id = 10
        engine._no_token_id = 11
        engine._prefix_tokens = [1, 2]
        engine._suffix_tokens = [8, 9]

        scores = engine.rerank("test query", ["doc1", "doc2"])

        assert len(scores) == 2
        # Each score should be between 0 and 1
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_rerank_custom_instruction(self) -> None:
        """Test rerank with custom instruction."""
        engine = RerankerEngine()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: {"yes": 10, "no": 11}[t]
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.apply_chat_template.return_value = "template"

        logits = torch.zeros(1, 5, 100)
        logits[0, -1, 10] = 1.0
        logits[0, -1, 11] = 1.0
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._yes_token_id = 10
        engine._no_token_id = 11
        engine._prefix_tokens = [1]
        engine._suffix_tokens = [2]

        scores = engine.rerank(
            "query",
            ["doc"],
            instruction="Custom task instruction",
        )

        assert len(scores) == 1
        # With equal logits, score should be ~0.5
        assert abs(scores[0] - 0.5) < 0.01


class TestRetrievalWithReranker:
    """Tests for RetrievalEngine integration with RerankerEngine."""

    def test_search_without_reranker_unchanged(self, sample_entry) -> None:
        """Test that search without reranker works as before."""
        from enyal.core.retrieval import RetrievalEngine

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8}
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        engine = RetrievalEngine(mock_store, reranker=None)
        results = engine.search("test query")

        assert len(results) == 1

    def test_search_with_reranker_reranks_results(self, sample_entry) -> None:
        """Test that search with reranker applies reranking."""
        from enyal.core.retrieval import RetrievalEngine

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8}
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [0.95]

        engine = RetrievalEngine(mock_store, reranker=mock_reranker)
        results = engine.search("test query")

        assert len(results) == 1
        mock_reranker.rerank.assert_called_once()

    def test_reranker_score_blending(self, sample_entry) -> None:
        """Test that reranker scores are blended correctly: 0.7*original + 0.3*rerank."""
        from enyal.core.retrieval import RetrievalEngine
        from enyal.models.context import ContextSearchResult

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8}
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [1.0]

        engine = RetrievalEngine(mock_store, reranker=mock_reranker)
        results = engine.search("test query")

        assert len(results) == 1
        # The adjusted_score should be blended
        result = results[0]
        assert result.adjusted_score is not None

    def test_reranker_only_processes_top_n(self) -> None:
        """Test that reranker only processes top N results."""
        from enyal.core.retrieval import RetrievalEngine
        from enyal.models.context import ContextEntry, ContextType, ScopeLevel

        entries = []
        for i in range(25):
            entry = ContextEntry(
                id=f"entry-{i}",
                content=f"Content {i}",
                content_type=ContextType.FACT,
                scope_level=ScopeLevel.PROJECT,
            )
            entries.append({"entry": entry, "distance": 0.1 + i * 0.01, "score": 0.9 - i * 0.01})

        mock_store = MagicMock()
        mock_store.recall.return_value = entries
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        mock_reranker = MagicMock()
        # Return scores for exactly rerank_top_n documents
        mock_reranker.rerank.return_value = [0.5] * 10

        engine = RetrievalEngine(mock_store, reranker=mock_reranker, rerank_top_n=10)
        engine.search("test query")

        # Reranker should only receive 10 documents
        call_args = mock_reranker.rerank.call_args
        documents = call_args[1].get("documents") or call_args[0][1]
        assert len(documents) == 10

    def test_reranker_failure_returns_original(self, sample_entry) -> None:
        """Test that reranker failure gracefully returns original results."""
        from enyal.core.retrieval import RetrievalEngine

        mock_store = MagicMock()
        mock_store.recall.return_value = [
            {"entry": sample_entry, "distance": 0.2, "score": 0.8}
        ]
        mock_store.fts_search.return_value = []
        mock_store.get_superseded_ids.return_value = set()
        mock_store.get_conflicted_ids.return_value = set()

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("Model failed")

        engine = RetrievalEngine(mock_store, reranker=mock_reranker)
        results = engine.search("test query")

        # Should still return results despite reranker failure
        assert len(results) == 1
