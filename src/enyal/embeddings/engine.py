"""Embedding engine for generating text embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from enyal.embeddings.models import ModelConfig

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
    Instance-based embedding engine using sentence-transformers.

    The model is loaded only when first needed, reducing cold start time
    for operations that don't require embeddings.

    Supports configurable models with prefix-based asymmetric embedding
    (different prefixes for documents vs queries).

    SSL Configuration:
        The engine automatically configures SSL settings from environment variables
        before downloading the model. Set these environment variables for corporate
        networks with SSL inspection:

        - ENYAL_SSL_CERT_FILE: Path to corporate CA certificate bundle
        - ENYAL_SSL_VERIFY: Set to "false" to disable verification (insecure)
        - ENYAL_MODEL_PATH: Path to pre-downloaded model directory
        - ENYAL_OFFLINE_MODE: Set to "true" to prevent network calls
        - ENYAL_HF_ENDPOINT: Custom HuggingFace Hub endpoint URL (e.g., Artifactory proxy)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the embedding engine.

        Args:
            config: Model configuration. If None, uses ModelConfig.from_env().
        """
        self._config = config or ModelConfig.from_env()
        self._model: SentenceTransformer | None = None

    @property
    def config(self) -> ModelConfig:
        """Return the model configuration."""
        return self._config

    def get_model(self) -> SentenceTransformer:
        """
        Get the sentence transformer model, loading it if necessary.

        The model will be downloaded from Hugging Face Hub on first use,
        unless ENYAL_MODEL_PATH or ENYAL_OFFLINE_MODE is set.

        If the initial load fails with an SSL error (common in corporate
        environments with SSL inspection proxies), the engine automatically
        disables SSL verification and retries.  This auto-recovery handles
        cases where:

        - ``ENYAL_SSL_VERIFY=false`` was not propagated to the MCP server
        - OpenSSL 3.x security level 2 rejects the corporate proxy's
          certificate during the TLS handshake (before cert verification)
        - The proxy CA has non-critical Basic Constraints or SHA-1 signatures

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            RuntimeError: If offline mode is enabled but model is not cached.
            Exception: If model loading fails for non-SSL reasons.
        """
        if self._model is None:
            # Configure SSL before importing sentence_transformers
            _ensure_ssl_configured()

            from enyal.core.ssl_config import get_model_path

            # Get model path (local or model name for download)
            model_path = get_model_path(self._config.name)

            logger.info(f"Loading embedding model: {model_path}")
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self._config.trust_remote_code:
                kwargs["trust_remote_code"] = True

            try:
                self._model = SentenceTransformer(model_path, **kwargs)
            except Exception as e:
                from enyal.core.ssl_config import _is_ssl_error

                if not _is_ssl_error(e):
                    raise

                # ── SSL Auto-Recovery ───────────────────────────────────
                # The initial load failed with an SSL error.  This is common
                # in corporate environments where:
                # 1. ENYAL_SSL_VERIFY was not passed to the MCP server process
                # 2. OpenSSL 3.x security level rejects the proxy's cert
                # 3. The proxy CA has non-critical Basic Constraints
                #
                # We automatically disable SSL at all layers and retry.
                logger.warning(
                    f"SSL error during model loading: {e}\n"
                    "Auto-recovering: disabling SSL verification and retrying.\n"
                    "To avoid this delay, set ENYAL_SSL_VERIFY=false in your "
                    "MCP server configuration's env block, or install the "
                    "'truststore' package (pip install truststore)."
                )

                from enyal.core.ssl_config import (
                    SSLConfig,
                    _disable_ssl_globally,
                    configure_http_backend,
                )

                _disable_ssl_globally()

                # Reconfigure HF Hub HTTP backend with verify=False
                ssl_override = SSLConfig(verify=False)
                configure_http_backend(ssl_override)

                # Clear any cached HF Hub sessions that used the old config
                try:
                    from huggingface_hub.utils._http import reset_sessions

                    reset_sessions()
                    logger.debug("Cleared huggingface_hub session cache for SSL retry")
                except (ImportError, AttributeError):
                    # Session cache API varies across HF Hub versions;
                    # the next request will create a new session from our
                    # factory regardless.
                    pass

                # Retry model loading
                self._model = SentenceTransformer(model_path, **kwargs)
                logger.warning(
                    "Model loaded successfully after SSL auto-recovery. "
                    "SSL verification is now disabled for this session."
                )

            logger.info("Embedding model loaded successfully")
        return self._model

    @staticmethod
    def _normalize(vec: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize a vector or batch of vectors to unit length.

        This ensures L2 distance approximates cosine distance (range 0-2),
        making the similarity formula 1/(1+d) produce meaningful scores.
        """
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        else:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid division by zero
            vec = vec / norms
        return vec.astype(np.float32)

    def embed(self, text: str) -> NDArray[np.float32]:
        """
        Generate embedding for a single text (for storage/indexing).

        Applies the document_prefix from config before encoding.
        Output is L2-normalized to unit length.

        Args:
            text: The text to embed.

        Returns:
            A unit-length float32 numpy array of shape (dimension,).
        """
        prefixed = f"{self._config.document_prefix}{text}"
        model = self.get_model()
        embedding: Any = model.encode(prefixed, convert_to_numpy=True)
        result: NDArray[np.float32] = embedding.astype(np.float32)
        return self._normalize(result)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """
        Generate embedding for a search query.

        Applies the query_prefix from config before encoding.
        Output is L2-normalized to unit length.

        Args:
            text: The query text to embed.

        Returns:
            A unit-length float32 numpy array of shape (dimension,).
        """
        prefixed = f"{self._config.query_prefix}{text}"
        model = self.get_model()
        embedding: Any = model.encode(prefixed, convert_to_numpy=True)
        result: NDArray[np.float32] = embedding.astype(np.float32)
        return self._normalize(result)

    def embed_batch(
        self,
        texts: list[str],
        task: str = "document",
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for multiple texts efficiently.

        Output vectors are L2-normalized to unit length.

        Args:
            texts: List of texts to embed.
            task: Either "document" (applies document_prefix) or "query" (applies query_prefix).
            batch_size: Number of texts to process at once.

        Returns:
            A (N, dimension) float32 numpy array of unit-length vectors.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._config.dimension)

        prefix = self._config.document_prefix if task == "document" else self._config.query_prefix
        prefixed = [f"{prefix}{t}" for t in texts]

        model = self.get_model()
        embeddings: Any = model.encode(
            prefixed,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        result: NDArray[np.float32] = embeddings.astype(np.float32)
        return self._normalize(result)

    def embedding_dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._config.dimension

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    def preload(self) -> None:
        """Preload the model (useful for warming up during startup)."""
        self.get_model()

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            logger.info("Unloading embedding model")
            self._model = None
