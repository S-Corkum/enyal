"""Embedding engine for generating text embeddings."""

import logging
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Lazy-loaded embedding engine using sentence-transformers.

    The model is loaded only when first needed, reducing cold start time
    for operations that don't require embeddings.
    """

    _model: ClassVar[object | None] = None
    _model_name: ClassVar[str] = "all-MiniLM-L6-v2"
    _embedding_dim: ClassVar[int] = 384

    @classmethod
    def get_model(cls) -> object:
        """
        Get the sentence transformer model, loading it if necessary.

        Returns:
            The loaded SentenceTransformer model.
        """
        if cls._model is None:
            logger.info(f"Loading embedding model: {cls._model_name}")
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(cls._model_name)
            logger.info("Embedding model loaded successfully")
        return cls._model

    @classmethod
    def embed(cls, text: str) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A 384-dimensional float32 numpy array.
        """
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)  # type: ignore[union-attr]
        return embedding.astype(np.float32)

    @classmethod
    def embed_batch(cls, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.

        Returns:
            A (N, 384) float32 numpy array where N is len(texts).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, cls._embedding_dim)

        model = cls.get_model()
        embeddings = model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)

    @classmethod
    def embedding_dimension(cls) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return cls._embedding_dim

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if the model is currently loaded."""
        return cls._model is not None

    @classmethod
    def preload(cls) -> None:
        """Preload the model (useful for warming up during startup)."""
        cls.get_model()

    @classmethod
    def unload(cls) -> None:
        """Unload the model to free memory."""
        if cls._model is not None:
            logger.info("Unloading embedding model")
            cls._model = None
