"""LLM wrappers: MLX (Apple Silicon) and Ollama (any OS).

This is the heart of the project: a thin adapter that exposes the
configured MLX model (`settings.mlx_model_repo`, default Qwen2.5-7B —
see PLAN.md §8 for why) as a LangChain-compatible chat model, via
mlx-lm. We use LangChain's
BaseChatModel interface so the LLM drops into the existing LangChain
ecosystem (chains, agents, RAGAS evaluation).

Hardware requirement for MLX: Apple Silicon (M1/M2/M3/M4/M5). On other
platforms the MLX constructor raises — no silent CPU fallback, no fake
responses. Set `LLM_PROVIDER=ollama` to use `OllamaChatModel` instead
(`get_chat_model()` picks the right one).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

from src.config import settings
from src.utils.logger import logger

# Mistral chat template markers (mlx-lm applies the chat template internally
# for instruct models that ship with a tokenizer.chat_template).
_MISTRAL_INSTRUCT_PROMPT = "<s>[INST] {system}\n\n{user} [/INST]"

# MLX's Metal command stream is bound to whichever OS thread first touches
# the GPU. LangChain's default async fallback (`BaseChatModel._agenerate`)
# runs our sync `_generate` via `run_in_executor(None, ...)` — the loop's
# *default* executor, which can hand the call to a different worker thread
# on every call. Each new thread crashes with `RuntimeError: There is no
# Stream(gpu, 0) in current thread.` (hit running RAGAS, which is fully
# async). Routing every MLX call through this single dedicated persistent
# thread instead keeps the Metal stream on one consistent thread for the
# life of the process, sync or async.
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-gpu")


def _find_earliest_stop(text: str, stop: list[str]) -> int | None:
    """Return the earliest index in `text` where any `stop` string starts."""
    positions = [text.index(s) for s in stop if s in text]
    return min(positions) if positions else None


def _truncate_at_stop(text: str, stop: list[str] | None) -> str:
    """Cut `text` at the first `stop` sequence, if any.

    mlx-lm's generate()/stream_generate() have no native stop-sequence
    support — without this, callers that rely on `stop` (e.g. LangChain's
    ReAct AgentExecutor, which stops generation at "\\nObservation:" so it
    can inject the real tool result) get a full uninterrupted completion:
    the model hallucinates its own Observation/Final Answer instead of
    ever actually being interrupted to call the tool.
    """
    if not stop:
        return text
    cut = _find_earliest_stop(text, stop)
    return text[:cut] if cut is not None else text


class MLXChatModel(BaseChatModel):
    """LangChain chat model backed by an MLX-quantized LLM.

    Lazy-loads mlx_lm on first call. Streaming is supported via the
    `stream` method (returns an iterator of strings).
    """

    # ClassVar (not pydantic private attrs): these are a genuine class-level
    # singleton cache shared across all instances. As bare `_name: Any`
    # annotations, pydantic v2 turns them into per-instance private attrs —
    # `MLXChatModel._model` then reads back a `ModelPrivateAttr` sentinel
    # instead of the real value, so `_load()`'s `is not None` guard never
    # fires the load, and separately a live `threading.Lock` default can't
    # be deepcopied when pydantic builds each instance's private attrs.
    # ClassVar opts these out of pydantic's model machinery entirely.
    _model: ClassVar[Any] = None
    _tokenizer: ClassVar[Any] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    model_name: str = settings.mlx_model_repo
    max_tokens: int = settings.mlx_max_tokens
    temperature: float = settings.mlx_temperature
    top_p: float = settings.mlx_top_p

    # Pydantic v2 config (was `class Config: arbitrary_types_allowed = True`
    # in v1 — deprecated since Pydantic 2.0, removed in 3.0).
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        """Flatten LangChain messages into a single chat-template prompt.

        If the tokenizer has a `chat_template` (virtually all modern
        instruct models do), we let it apply the model's own format via
        `apply_chat_template` — model-agnostic, works for ChatML (Qwen),
        Mistral's [INST] format, Llama, etc. Only build a flat string
        manually if no template is available at all.

        Some chat templates (notably Mistral's) have no "system" role and
        enforce strict user/assistant alternation — passing a separate
        "system" turn raises `Conversation roles must alternate
        user/assistant/...`. We try the native roles first (so models
        that DO support a system role, like Qwen's ChatML, get it) and
        only fold system into the first user turn on that specific
        failure.
        """
        self._load()
        tokenizer = MLXChatModel._tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            native_chat = [
                {
                    "role": "system"
                    if isinstance(m, SystemMessage)
                    else "assistant"
                    if isinstance(m, AIMessage)
                    else "user",
                    "content": m.content,
                }
                for m in messages
            ]
            try:
                return tokenizer.apply_chat_template(
                    native_chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.debug(
                    "Chat template rejected a separate system role ({}); "
                    "folding it into the first user turn instead.",
                    e,
                )
            system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
            chat: list[dict[str, str]] = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    continue
                role = "assistant" if isinstance(m, AIMessage) else "user"
                content = m.content
                if system and role == "user" and not chat:
                    content = f"{system}\n\n{content}"
                    system = ""
                chat.append({"role": role, "content": content})
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: naive concatenation (only if the tokenizer ships no
        # chat template at all — not expected for either Qwen or Mistral).
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        user = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))
        return _MISTRAL_INSTRUCT_PROMPT.format(system=system, user=user)

    def _make_sampler(self):
        """Build the mlx-lm sampler for this model's temp/top_p.

        mlx-lm >=0.24 dropped the `temp`/`top_p` kwargs from
        `generate`/`generate_step` in favor of a `sampler` callable built
        via `mlx_lm.sample_utils.make_sampler`.
        """
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=self.temperature, top_p=self.top_p)

    def _generate_impl(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> ChatResult:
        """The actual (blocking) generation. Must only ever run on
        `_MLX_EXECUTOR`'s dedicated thread — see that constant's docstring."""
        from mlx_lm import generate

        prompt = self._messages_to_prompt(messages)
        response = generate(
            MLXChatModel._model,
            MLXChatModel._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self._make_sampler(),
            verbose=False,
        )
        text = response if isinstance(response, str) else str(response)
        text = _truncate_at_stop(text, stop).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # `run_manager` (LangChain's callback manager) is part of the
        # standard BaseChatModel._generate signature and some callers
        # (e.g. RAGAS's LangchainLLMWrapper) pass it positionally — without
        # this parameter, that raises `takes from 2 to 3 positional
        # arguments but 4 were given`. We don't emit callbacks, so it's
        # accepted and ignored.
        #
        # Dispatched through `_MLX_EXECUTOR` (not run inline on the
        # calling thread) so this stays on the same thread as `_agenerate`
        # and `_stream`/`_astream` — see `_MLX_EXECUTOR`'s docstring.
        return _MLX_EXECUTOR.submit(self._generate_impl, messages, stop).result()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_MLX_EXECUTOR, self._generate_impl, messages, stop)

    def _stream_impl(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        out_queue: Any,
    ) -> None:
        """The actual (blocking) streaming loop. Must only ever run on
        `_MLX_EXECUTOR`'s dedicated thread. Pushes chunks (or `None` as
        end-of-stream sentinel) onto `out_queue` so a caller on a
        *different* thread can pull them without itself touching MLX."""
        from mlx_lm import stream_generate

        prompt = self._messages_to_prompt(messages)
        buffer = ""
        try:
            for chunk in stream_generate(
                MLXChatModel._model,
                MLXChatModel._tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=self._make_sampler(),
            ):
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                if not text:
                    continue
                prev_len = len(buffer)
                buffer += text
                if stop:
                    cut = _find_earliest_stop(buffer, stop)
                    if cut is not None:
                        keep = text[: max(cut - prev_len, 0)]
                        if keep:
                            out_queue.put(keep)
                        return
                out_queue.put(text)
        finally:
            out_queue.put(None)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import queue as _queue

        out_queue: _queue.Queue = _queue.Queue()
        _MLX_EXECUTOR.submit(self._stream_impl, messages, stop, out_queue)
        while True:
            text = out_queue.get()
            if text is None:
                return
            # Must be a *Chunk type: langchain merges consecutive stream
            # chunks with `+`, which plain ChatGeneration/AIMessage don't
            # support (raises `unsupported operand type(s) for +`).
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

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


def _role(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, AIMessage):
        return "assistant"
    return "user"


class OllamaChatModel(BaseChatModel):
    """LangChain chat model backed by a local Ollama server (/api/chat).

    The portable counterpart of `MLXChatModel`: same LangChain interface,
    same sampling settings, but the GPU work happens inside Ollama (CUDA,
    ROCm, Metal or CPU — whatever the host has). Talks plain HTTP with
    httpx, so no extra dependency. Token counts reported by Ollama are
    kept on the AIMessage (`usage_metadata`) for the LLMOps bench and the
    API metrics.
    """

    model_name: str = settings.ollama_chat_model
    base_url: str = settings.ollama_base_url
    max_tokens: int = settings.mlx_max_tokens
    temperature: float = settings.mlx_temperature
    top_p: float = settings.mlx_top_p
    num_ctx: int = settings.ollama_num_ctx
    timeout: float = settings.ollama_timeout_seconds

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _payload(self, messages: list[BaseMessage], stop: list[str] | None, stream: bool) -> dict:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if stop:
            options["stop"] = stop
        return {
            "model": self.model_name,
            "messages": [{"role": _role(m), "content": m.content} for m in messages],
            "stream": stream,
            # Reasoning models (qwen3.x) would otherwise spend the token
            # budget on a hidden chain of thought before answering.
            "think": False,
            "keep_alive": "10m",
            "options": options,
        }

    @staticmethod
    def _usage(data: dict) -> dict[str, int]:
        prompt = int(data.get("prompt_eval_count") or 0)
        completion = int(data.get("eval_count") or 0)
        return {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json=self._payload(messages, stop, stream=False),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = _truncate_at_stop(data["message"]["content"], stop).strip()
        message = AIMessage(content=text, usage_metadata=self._usage(data))
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"token_usage": self._usage(data), "model_name": self.model_name},
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import json

        import httpx

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(messages, stop, stream=True),
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                text = data.get("message", {}).get("content", "")
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                if data.get("done"):
                    return

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


def get_chat_model() -> BaseChatModel:
    """Return the chat model selected by `LLM_PROVIDER` (mlx | ollama)."""
    settings.assert_backend_supported()
    if settings.llm_provider == "ollama":
        return OllamaChatModel()
    return MLXChatModel()


__all__ = ["MLXChatModel", "OllamaChatModel", "get_chat_model"]
