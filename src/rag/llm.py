"""MLX LLM wrapper.

This is the heart of the project: a thin adapter that exposes Mistral 7B
(via mlx-lm) as a LangChain-compatible chat model. We use LangChain's
BaseChatModel interface so the LLM drops into the existing LangChain
ecosystem (chains, agents, RAGAS evaluation).

Hardware requirement: Apple Silicon (M1/M2/M3/M4/M5). On other platforms
the constructor raises — no silent CPU fallback, no fake responses.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.config import settings
from src.utils.logger import logger

# Mistral chat template markers (mlx-lm applies the chat template internally
# for instruct models that ship with a tokenizer.chat_template).
_MISTRAL_INSTRUCT_PROMPT = "<s>[INST] {system}\n\n{user} [/INST]"


class MLXChatModel(BaseChatModel):
    """LangChain chat model backed by an MLX-quantized LLM.

    Lazy-loads mlx_lm on first call. Streaming is supported via the
    `stream` method (returns an iterator of strings).
    """

    _model: Any = None
    _tokenizer: Any = None
    _lock: threading.Lock = threading.Lock()

    model_name: str = settings.mlx_model_repo
    max_tokens: int = settings.mlx_max_tokens
    temperature: float = settings.mlx_temperature
    top_p: float = settings.mlx_top_p

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Fail fast on non-Apple-Silicon — no fake fallback.
        settings.assert_apple_silicon()

    def _load(self) -> None:
        if MLXChatModel._model is not None:
            return
        with MLXChatModel._lock:
            if MLXChatModel._model is not None:
                return
            from mlx_lm import load

            logger.info("Loading MLX model {} (this takes ~10 sec on M-series)", self.model_name)
            model, tokenizer = load(self.model_name)
            MLXChatModel._model = model
            MLXChatModel._tokenizer = tokenizer
            logger.info("MLX model ready")

    def _messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        """Flatten LangChain messages into a single Mistral instruct prompt.

        We use the simple [INST] template. If the tokenizer has a
        `chat_template` attribute (newer Mistral variants), mlx-lm will
        apply it via `apply_chat_template` — we let it do that and
        only build a flat string if no template is available.
        """
        self._load()
        tokenizer = MLXChatModel._tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system" if isinstance(m, SystemMessage) else "user",
                     "content": m.content}
                    for m in messages
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: naive concatenation
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        user = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))
        return _MISTRAL_INSTRUCT_PROMPT.format(system=system, user=user)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        from mlx_lm import generate

        prompt = self._messages_to_prompt(messages)
        response = generate(
            MLXChatModel._model,
            MLXChatModel._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            top_p=self.top_p,
            verbose=False,
        )
        text = response.strip() if isinstance(response, str) else str(response).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        from mlx_lm import stream_generate

        prompt = self._messages_to_prompt(messages)
        for chunk in stream_generate(
            MLXChatModel._model,
            MLXChatModel._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            top_p=self.top_p,
        ):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            yield ChatGeneration(message=AIMessage(content=text))

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


__all__ = ["MLXChatModel"]
