"""Enyal: Persistent, queryable memory for AI coding agents."""

__version__ = "0.1.0"

from enyal.core.retrieval import RetrievalEngine
from enyal.core.store import ContextStore
from enyal.models.context import ContextEntry, ContextType, ScopeLevel

__all__ = [
    "ContextStore",
    "RetrievalEngine",
    "ContextEntry",
    "ContextType",
    "ScopeLevel",
    "__version__",
]
