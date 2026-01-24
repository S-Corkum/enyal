"""Local embedding generation."""

from enyal.embeddings.engine import EmbeddingEngine
from enyal.embeddings.models import MODEL_REGISTRY, ModelConfig

__all__ = ["EmbeddingEngine", "ModelConfig", "MODEL_REGISTRY"]
