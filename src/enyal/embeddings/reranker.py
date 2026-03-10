"""Reranker engine using Qwen3-Reranker-0.6B for post-retrieval reranking."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the Query may be truncated for privacy or "
    "other reasons, do not penalize for this. Reply only with 'yes' or 'no'."
)


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for the reranker model.

    Attributes:
        name: HuggingFace model name or path.
        max_length: Maximum input sequence length in tokens.
        trust_remote_code: Whether to trust remote code for model loading.
        instruction: Default task instruction for reranking.
    """

    name: str = "Qwen/Qwen3-Reranker-0.6B"
    max_length: int = 8192
    trust_remote_code: bool = True
    instruction: str = field(default=_DEFAULT_INSTRUCTION)

    @classmethod
    def from_env(cls) -> RerankerConfig:
        """Create a RerankerConfig from environment variables.

        Reads:
            ENYAL_RERANKER_MODEL: Model name (default: Qwen/Qwen3-Reranker-0.6B)

        Returns:
            RerankerConfig for the specified model.
        """
        model_name = os.environ.get("ENYAL_RERANKER_MODEL", "")
        if model_name:
            return RerankerConfig(name=model_name)
        return RerankerConfig()


class RerankerEngine:
    """Reranker engine using causal LM yes/no logit scoring.

    The model is loaded only when first needed, reducing memory usage
    when reranking is not required.
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        """Initialize the reranker engine.

        Args:
            config: Reranker configuration. If None, uses RerankerConfig.from_env().
        """
        self._config = config or RerankerConfig.from_env()
        self._model: Any = None
        self._tokenizer: Any = None
        self._yes_token_id: int | None = None
        self._no_token_id: int | None = None
        self._prefix_tokens: Any = None
        self._suffix_tokens: Any = None

    @property
    def config(self) -> RerankerConfig:
        """Return the reranker configuration."""
        return self._config

    def _get_model(self) -> tuple[Any, Any]:
        """Load the reranker model and tokenizer.

        Returns:
            Tuple of (model, tokenizer).
        """
        if self._model is not None:
            return self._model, self._tokenizer

        from enyal.embeddings.engine import _ensure_ssl_configured

        _ensure_ssl_configured()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading reranker model: {self._config.name}")

        model_kwargs: dict[str, Any] = {}
        if self._config.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        # Workaround: Qwen3 SDPA produces NaN on macOS (Intel + Apple Silicon)
        # https://github.com/huggingface/sentence-transformers/issues/3498
        if sys.platform == "darwin":
            model_kwargs["attn_implementation"] = "eager"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.name,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.name,
            torch_dtype=torch.float32,
            **model_kwargs,
        )
        self._model.eval()

        # Cache yes/no token IDs
        self._yes_token_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_token_id = self._tokenizer.convert_tokens_to_ids("no")

        # Pre-compute prefix/suffix tokens for the chat template
        prefix = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYSTEM_PROMPT}],
            tokenize=False,
            add_generation_prompt=False,
        )
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)

        suffix_text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": "placeholder"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Extract only the part after "placeholder"
        suffix_only = suffix_text.split("placeholder")[-1]
        self._suffix_tokens = self._tokenizer.encode(suffix_only, add_special_tokens=False)

        logger.info("Reranker model loaded successfully")
        return self._model, self._tokenizer

    @staticmethod
    def _format_input(query: str, document: str, instruction: str) -> str:
        """Format a query-document pair for scoring.

        Args:
            query: The search query.
            document: The document to score.
            instruction: The task instruction.

        Returns:
            Formatted input string.
        """
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """Score documents against a query using yes/no logit probabilities.

        Args:
            query: The search query.
            documents: List of document texts to score.
            instruction: Optional task instruction override.

        Returns:
            List of relevance scores (0-1), one per document.
        """
        if not documents:
            return []

        import torch

        model, tokenizer = self._get_model()
        task_instruction = instruction or self._config.instruction

        scores: list[float] = []
        with torch.no_grad():
            for doc in documents:
                formatted = self._format_input(query, doc, task_instruction)
                user_tokens = tokenizer.encode(formatted, add_special_tokens=False)

                # Build input: prefix + user content + suffix
                input_ids = self._prefix_tokens + user_tokens + self._suffix_tokens
                input_tensor = torch.tensor([input_ids], dtype=torch.long)

                # Truncate to max length
                if input_tensor.shape[1] > self._config.max_length:
                    input_tensor = input_tensor[:, : self._config.max_length]

                outputs = model(input_ids=input_tensor)
                logits = outputs.logits[0, -1, :]

                # Extract yes/no logits and compute probability
                yes_logit = logits[self._yes_token_id]
                no_logit = logits[self._no_token_id]
                probs = torch.softmax(
                    torch.stack([yes_logit, no_logit]), dim=0
                )
                score = probs[0].item()  # P(yes)
                scores.append(score)

        return scores

    def unload(self) -> None:
        """Unload the model and tokenizer to free memory."""
        if self._model is not None:
            logger.info("Unloading reranker model")
            self._model = None
            self._tokenizer = None
            self._yes_token_id = None
            self._no_token_id = None
            self._prefix_tokens = None
            self._suffix_tokens = None

    def is_loaded(self) -> bool:
        """Check if the reranker model is currently loaded."""
        return self._model is not None
