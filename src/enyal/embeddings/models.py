"""Embedding model configuration and registry."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an embedding model.

    Attributes:
        name: HuggingFace model name or path.
        dimension: Output embedding dimensionality.
        document_prefix: Prefix prepended to texts for storage/indexing.
        query_prefix: Prefix prepended to texts for search queries.
        max_seq_length: Maximum input sequence length in tokens.
        trust_remote_code: Whether to trust remote code for model loading.
    """

    name: str
    dimension: int
    document_prefix: str = ""
    query_prefix: str = ""
    max_seq_length: int = 512
    trust_remote_code: bool = False

    @classmethod
    def default(cls) -> ModelConfig:
        """Return the default model configuration (nomic-embed-text-v1.5)."""
        return MODEL_REGISTRY["nomic-ai/nomic-embed-text-v1.5"]

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Create a ModelConfig from environment variables.

        Reads:
            ENYAL_MODEL_NAME: Model name (default: nomic-ai/nomic-embed-text-v1.5)
            ENYAL_MODEL_DIMENSION: Override dimension for custom models
            ENYAL_TRUST_REMOTE_CODE: Override trust_remote_code (true/false)

        Returns:
            ModelConfig for the specified model.
        """
        model_name = os.environ.get("ENYAL_MODEL_NAME", "")

        if not model_name:
            return cls.default()

        # Check registry first
        if model_name in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model_name]
            # Allow env override of trust_remote_code
            trust_override = os.environ.get("ENYAL_TRUST_REMOTE_CODE", "")
            if trust_override:
                return ModelConfig(
                    name=config.name,
                    dimension=config.dimension,
                    document_prefix=config.document_prefix,
                    query_prefix=config.query_prefix,
                    max_seq_length=config.max_seq_length,
                    trust_remote_code=trust_override.lower() == "true",
                )
            return config

        # Custom model - require dimension
        dimension_str = os.environ.get("ENYAL_MODEL_DIMENSION", "")
        if not dimension_str:
            raise ValueError(
                f"Custom model '{model_name}' requires ENYAL_MODEL_DIMENSION to be set. "
                f"Known models: {', '.join(MODEL_REGISTRY.keys())}"
            )

        trust_remote = os.environ.get("ENYAL_TRUST_REMOTE_CODE", "false").lower() == "true"

        return ModelConfig(
            name=model_name,
            dimension=int(dimension_str),
            trust_remote_code=trust_remote,
        )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "nomic-ai/nomic-embed-text-v1.5": ModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        max_seq_length=8192,
        trust_remote_code=True,
    ),
    "all-MiniLM-L6-v2": ModelConfig(
        name="all-MiniLM-L6-v2",
        dimension=384,
        document_prefix="",
        query_prefix="",
        max_seq_length=256,
        trust_remote_code=False,
    ),
    "intfloat/e5-small-v2": ModelConfig(
        name="intfloat/e5-small-v2",
        dimension=384,
        document_prefix="passage: ",
        query_prefix="query: ",
        max_seq_length=512,
        trust_remote_code=False,
    ),
    "BAAI/bge-small-en-v1.5": ModelConfig(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        document_prefix="",
        query_prefix="Represent this sentence for searching relevant passages: ",
        max_seq_length=512,
        trust_remote_code=False,
    ),
}
