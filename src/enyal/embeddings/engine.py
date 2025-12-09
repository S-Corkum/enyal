"""Embedding engine for generating text embeddings."""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Flag to track if SSL has been configured
_ssl_configured: bool = False


def _ensure_ssl_configured() -> None:
    """
    Ensure SSL is configured before model download.

    This must be called before importing sentence_transformers for the first time,
    as the configuration affects how the library makes HTTP requests.
    """
    global _ssl_configured
    if _ssl_configured:
        return

    from enyal.core.ssl_config import (
        configure_http_backend,
        configure_ssl_environment,
        get_ssl_config,
    )

    config = get_ssl_config()
    configure_ssl_environment(config)
    configure_http_backend(config)
    _ssl_configured = True
    logger.debug("SSL configuration applied for embedding model download")


class EmbeddingEngine:
    """
    Lazy-loaded embedding engine using sentence-transformers.

    The model is loaded only when first needed, reducing cold start time
    for operations that don't require embeddings.

    SSL Configuration:
        The engine automatically configures SSL settings from environment variables
        before downloading the model. Set these environment variables for corporate
        networks with SSL inspection:

        - ENYAL_SSL_CERT_FILE: Path to corporate CA certificate bundle
        - ENYAL_SSL_VERIFY: Set to "false" to disable verification (insecure)
        - ENYAL_MODEL_PATH: Path to pre-downloaded model directory
        - ENYAL_OFFLINE_MODE: Set to "true" to prevent network calls
    """

    _model: ClassVar["SentenceTransformer | None"] = None
    _model_name: ClassVar[str] = "all-MiniLM-L6-v2"
    _embedding_dim: ClassVar[int] = 384

    @classmethod
    def get_model(cls) -> "SentenceTransformer":
        """
        Get the sentence transformer model, loading it if necessary.

        The model will be downloaded from Hugging Face Hub on first use,
        unless ENYAL_MODEL_PATH or ENYAL_OFFLINE_MODE is set.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            SSLError: If SSL verification fails in corporate environments.
                Configure ENYAL_SSL_CERT_FILE with your CA bundle.
            RuntimeError: If offline mode is enabled but model is not cached.
        """
        if cls._model is None:
            # Configure SSL before importing sentence_transformers
            _ensure_ssl_configured()

            from enyal.core.ssl_config import get_model_path

            # Get model path (local or model name for download)
            model_path = get_model_path(cls._model_name)

            logger.info(f"Loading embedding model: {model_path}")
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(model_path)
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
        embedding: Any = model.encode(text, convert_to_numpy=True)
        result: NDArray[np.float32] = embedding.astype(np.float32)
        return result

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
        embeddings: Any = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        result: NDArray[np.float32] = embeddings.astype(np.float32)
        return result

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
