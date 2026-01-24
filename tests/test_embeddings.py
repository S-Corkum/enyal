"""Tests for embedding engine and model configuration."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enyal.embeddings.engine import EmbeddingEngine
from enyal.embeddings.models import MODEL_REGISTRY, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass and registry."""

    def test_default_returns_nomic(self) -> None:
        """Test that default() returns nomic-embed-text-v1.5 config."""
        config = ModelConfig.default()
        assert config.name == "nomic-ai/nomic-embed-text-v1.5"
        assert config.dimension == 768
        assert config.trust_remote_code is True

    def test_default_has_prefixes(self) -> None:
        """Test that default config has search prefixes."""
        config = ModelConfig.default()
        assert config.document_prefix == "search_document: "
        assert config.query_prefix == "search_query: "

    def test_from_env_no_var_returns_default(self) -> None:
        """Test from_env with no ENYAL_MODEL_NAME returns default."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ENYAL_MODEL_NAME if present
            os.environ.pop("ENYAL_MODEL_NAME", None)
            config = ModelConfig.from_env()
            assert config.name == "nomic-ai/nomic-embed-text-v1.5"

    def test_from_env_known_model(self) -> None:
        """Test from_env with a known model name."""
        with patch.dict(os.environ, {"ENYAL_MODEL_NAME": "all-MiniLM-L6-v2"}):
            config = ModelConfig.from_env()
            assert config.name == "all-MiniLM-L6-v2"
            assert config.dimension == 384
            assert config.trust_remote_code is False

    def test_from_env_custom_model_with_dimension(self) -> None:
        """Test from_env with a custom model and dimension."""
        with patch.dict(
            os.environ,
            {"ENYAL_MODEL_NAME": "custom/model", "ENYAL_MODEL_DIMENSION": "512"},
        ):
            config = ModelConfig.from_env()
            assert config.name == "custom/model"
            assert config.dimension == 512
            assert config.trust_remote_code is False

    def test_from_env_custom_model_without_dimension_raises(self) -> None:
        """Test from_env with custom model but no dimension raises ValueError."""
        with patch.dict(os.environ, {"ENYAL_MODEL_NAME": "custom/model"}):
            os.environ.pop("ENYAL_MODEL_DIMENSION", None)
            with pytest.raises(ValueError, match="requires ENYAL_MODEL_DIMENSION"):
                ModelConfig.from_env()

    def test_from_env_trust_remote_code_override(self) -> None:
        """Test from_env with ENYAL_TRUST_REMOTE_CODE override."""
        with patch.dict(
            os.environ,
            {
                "ENYAL_MODEL_NAME": "nomic-ai/nomic-embed-text-v1.5",
                "ENYAL_TRUST_REMOTE_CODE": "false",
            },
        ):
            config = ModelConfig.from_env()
            assert config.trust_remote_code is False

    def test_registry_contains_known_models(self) -> None:
        """Test that MODEL_REGISTRY has all expected models."""
        assert "nomic-ai/nomic-embed-text-v1.5" in MODEL_REGISTRY
        assert "all-MiniLM-L6-v2" in MODEL_REGISTRY
        assert "intfloat/e5-small-v2" in MODEL_REGISTRY
        assert "BAAI/bge-small-en-v1.5" in MODEL_REGISTRY

    def test_registry_nomic_config(self) -> None:
        """Test nomic registry entry."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        assert config.dimension == 768
        assert config.max_seq_length == 8192
        assert config.trust_remote_code is True

    def test_registry_e5_config(self) -> None:
        """Test e5-small-v2 registry entry."""
        config = MODEL_REGISTRY["intfloat/e5-small-v2"]
        assert config.dimension == 384
        assert config.document_prefix == "passage: "
        assert config.query_prefix == "query: "

    def test_registry_bge_config(self) -> None:
        """Test bge-small-en-v1.5 registry entry."""
        config = MODEL_REGISTRY["BAAI/bge-small-en-v1.5"]
        assert config.dimension == 384
        assert config.document_prefix == ""
        assert "Represent this sentence" in config.query_prefix

    def test_frozen_dataclass(self) -> None:
        """Test that ModelConfig is immutable."""
        config = ModelConfig.default()
        with pytest.raises(AttributeError):
            config.name = "something-else"  # type: ignore[misc]


class TestEmbeddingEngine:
    """Tests for the instance-based EmbeddingEngine."""

    def test_instance_creation_with_config(self) -> None:
        """Test creating an engine with explicit config."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)
        assert engine.config == config
        assert engine.config.dimension == 384

    def test_instance_creation_default_config(self) -> None:
        """Test creating an engine with default config from env."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENYAL_MODEL_NAME", None)
            engine = EmbeddingEngine()
            assert engine.config.name == "nomic-ai/nomic-embed-text-v1.5"

    def test_embed_applies_document_prefix(self) -> None:
        """Test that embed prepends document_prefix."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        engine._model = mock_model

        engine.embed("test text")

        # Verify prefix was applied
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == "search_document: test text"

    def test_embed_query_applies_query_prefix(self) -> None:
        """Test that embed_query prepends query_prefix."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
        engine._model = mock_model

        engine.embed_query("search query")

        # Verify prefix was applied
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == "search_query: search query"

    def test_embed_no_prefix_when_empty(self) -> None:
        """Test that embed works with empty prefix."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        engine._model = mock_model

        engine.embed("test text")

        call_args = mock_model.encode.call_args[0][0]
        assert call_args == "test text"

    def test_embed_batch_document_task(self) -> None:
        """Test batch embedding with document task."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)
        engine._model = mock_model

        texts = ["one", "two", "three"]
        result = engine.embed_batch(texts, task="document")

        assert result.shape == (3, 768)
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == "search_document: one"

    def test_embed_batch_query_task(self) -> None:
        """Test batch embedding with query task."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)
        engine._model = mock_model

        texts = ["query one", "query two"]
        result = engine.embed_batch(texts, task="query")

        assert result.shape == (2, 768)
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == "search_query: query one"

    def test_embed_batch_empty(self) -> None:
        """Test batch embedding with empty list."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        result = engine.embed_batch([])
        assert result.shape == (0, 768)

    def test_embedding_dimension(self) -> None:
        """Test embedding_dimension returns config value."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)
        assert engine.embedding_dimension() == 384

        config768 = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine768 = EmbeddingEngine(config768)
        assert engine768.embedding_dimension() == 768

    def test_is_loaded_false(self) -> None:
        """Test is_loaded returns False when model is not loaded."""
        engine = EmbeddingEngine(MODEL_REGISTRY["all-MiniLM-L6-v2"])
        assert engine.is_loaded() is False

    def test_is_loaded_true(self) -> None:
        """Test is_loaded returns True when model is loaded."""
        engine = EmbeddingEngine(MODEL_REGISTRY["all-MiniLM-L6-v2"])
        engine._model = MagicMock()
        assert engine.is_loaded() is True

    def test_unload(self) -> None:
        """Test unload clears the model."""
        engine = EmbeddingEngine(MODEL_REGISTRY["all-MiniLM-L6-v2"])
        engine._model = MagicMock()

        engine.unload()
        assert engine._model is None

    def test_get_model_lazy_loading(self) -> None:
        """Test that get_model loads model lazily."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        with (
            patch("enyal.embeddings.engine._ensure_ssl_configured"),
            patch("enyal.core.ssl_config.get_model_path", return_value="all-MiniLM-L6-v2"),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            model1 = engine.get_model()
            assert mock_st.call_count == 1

            model2 = engine.get_model()
            assert mock_st.call_count == 1  # Not called again

            assert model1 is model2

    def test_get_model_trust_remote_code(self) -> None:
        """Test that trust_remote_code is passed to SentenceTransformer."""
        config = MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]
        engine = EmbeddingEngine(config)

        with (
            patch("enyal.embeddings.engine._ensure_ssl_configured"),
            patch(
                "enyal.core.ssl_config.get_model_path",
                return_value="nomic-ai/nomic-embed-text-v1.5",
            ),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = MagicMock()
            engine.get_model()

            mock_st.assert_called_once_with(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
            )

    def test_get_model_no_trust_remote_code(self) -> None:
        """Test that trust_remote_code is not passed when False."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        with (
            patch("enyal.embeddings.engine._ensure_ssl_configured"),
            patch("enyal.core.ssl_config.get_model_path", return_value="all-MiniLM-L6-v2"),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):
            mock_st.return_value = MagicMock()
            engine.get_model()

            # Should be called without trust_remote_code kwarg
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    def test_preload_calls_get_model(self) -> None:
        """Test that preload calls get_model."""
        engine = EmbeddingEngine(MODEL_REGISTRY["all-MiniLM-L6-v2"])

        with patch.object(engine, "get_model") as mock_get:
            engine.preload()
            mock_get.assert_called_once()

    def test_embed_returns_float32(self) -> None:
        """Test that embed always returns float32 dtype."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float64)
        engine._model = mock_model

        result = engine.embed("test")
        assert result.dtype == np.float32

    def test_embed_batch_shows_progress_for_large_batches(self) -> None:
        """Test that embed_batch shows progress for large batches."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(150, 384).astype(np.float32)
        engine._model = mock_model

        texts = ["text"] * 150
        engine.embed_batch(texts)

        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("show_progress_bar") is True

    def test_embed_batch_no_progress_for_small_batches(self) -> None:
        """Test that embed_batch doesn't show progress for small batches."""
        config = MODEL_REGISTRY["all-MiniLM-L6-v2"]
        engine = EmbeddingEngine(config)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(50, 384).astype(np.float32)
        engine._model = mock_model

        texts = ["text"] * 50
        engine.embed_batch(texts)

        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("show_progress_bar") is False
