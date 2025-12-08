"""Tests for embedding engine module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from enyal.embeddings.engine import EmbeddingEngine


@pytest.fixture(autouse=True)
def reset_model() -> None:
    """Reset the model class variable between tests."""
    EmbeddingEngine._model = None


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine class."""

    def test_embed_single_text(self) -> None:
        """Test embedding a single text string."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_get_model.return_value = mock_model

            result = EmbeddingEngine.embed("test text")

            assert result.shape == (384,)
            assert result.dtype == np.float32
            mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    def test_embed_batch_empty_list(self) -> None:
        """Test embedding an empty list."""
        result = EmbeddingEngine.embed_batch([])

        assert result.shape == (0, 384)
        assert result.dtype == np.float32

    def test_embed_batch_single_item(self) -> None:
        """Test embedding a batch with a single item."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
            mock_get_model.return_value = mock_model

            result = EmbeddingEngine.embed_batch(["single text"])

            assert result.shape == (1, 384)
            assert result.dtype == np.float32

    def test_embed_batch_multiple_items(self) -> None:
        """Test embedding a batch with multiple items."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
            mock_get_model.return_value = mock_model

            texts = ["text one", "text two", "text three"]
            result = EmbeddingEngine.embed_batch(texts)

            assert result.shape == (3, 384)
            assert result.dtype == np.float32

    def test_embed_batch_custom_batch_size(self) -> None:
        """Test embedding with custom batch size."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)
            mock_get_model.return_value = mock_model

            texts = ["text"] * 5
            result = EmbeddingEngine.embed_batch(texts, batch_size=16)

            assert result.shape == (5, 384)
            # Verify batch_size was passed
            mock_model.encode.assert_called_once()
            call_kwargs = mock_model.encode.call_args.kwargs
            assert call_kwargs.get("batch_size") == 16

    def test_embedding_dimension(self) -> None:
        """Test embedding_dimension returns correct value."""
        dim = EmbeddingEngine.embedding_dimension()

        assert dim == 384

    def test_is_loaded_false(self) -> None:
        """Test is_loaded returns False when model is not loaded."""
        EmbeddingEngine._model = None

        assert EmbeddingEngine.is_loaded() is False

    def test_is_loaded_true(self) -> None:
        """Test is_loaded returns True when model is loaded."""
        EmbeddingEngine._model = MagicMock()

        assert EmbeddingEngine.is_loaded() is True

    def test_preload(self) -> None:
        """Test preload loads the model."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            EmbeddingEngine.preload()

            mock_st.assert_called_once_with("all-MiniLM-L6-v2")
            assert EmbeddingEngine._model is not None

    def test_unload(self) -> None:
        """Test unload clears the model."""
        EmbeddingEngine._model = MagicMock()

        EmbeddingEngine.unload()

        assert EmbeddingEngine._model is None

    def test_get_model_lazy_loading(self) -> None:
        """Test that get_model loads model lazily."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            # First call should load
            model1 = EmbeddingEngine.get_model()
            assert mock_st.call_count == 1

            # Second call should return cached
            model2 = EmbeddingEngine.get_model()
            assert mock_st.call_count == 1  # Still 1, not called again

            assert model1 is model2

    def test_embed_uses_get_model(self) -> None:
        """Test that embed uses get_model method."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_get_model.return_value = mock_model

            EmbeddingEngine.embed("test")

            mock_get_model.assert_called_once()

    def test_embed_batch_shows_progress_for_large_batches(self) -> None:
        """Test that embed_batch shows progress for large batches."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(150, 384).astype(np.float32)
            mock_get_model.return_value = mock_model

            texts = ["text"] * 150  # More than 100 items
            EmbeddingEngine.embed_batch(texts)

            # Check that show_progress_bar=True was passed
            call_kwargs = mock_model.encode.call_args.kwargs
            assert call_kwargs.get("show_progress_bar") is True

    def test_embed_batch_no_progress_for_small_batches(self) -> None:
        """Test that embed_batch doesn't show progress for small batches."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(50, 384).astype(np.float32)
            mock_get_model.return_value = mock_model

            texts = ["text"] * 50  # Less than 100 items
            EmbeddingEngine.embed_batch(texts)

            # Check that show_progress_bar=False was passed
            call_kwargs = mock_model.encode.call_args.kwargs
            assert call_kwargs.get("show_progress_bar") is False

    def test_embed_returns_float32(self) -> None:
        """Test that embed always returns float32 dtype."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return float64 to test conversion
            mock_model.encode.return_value = np.random.rand(384).astype(np.float64)
            mock_get_model.return_value = mock_model

            result = EmbeddingEngine.embed("test")

            assert result.dtype == np.float32

    def test_embed_batch_returns_float32(self) -> None:
        """Test that embed_batch always returns float32 dtype."""
        with patch.object(EmbeddingEngine, "get_model") as mock_get_model:
            mock_model = MagicMock()
            # Return float64 to test conversion
            mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float64)
            mock_get_model.return_value = mock_model

            result = EmbeddingEngine.embed_batch(["text"] * 5)

            assert result.dtype == np.float32
