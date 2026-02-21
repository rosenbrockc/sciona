"""LLM client abstraction for the Hunter agent."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM completion."""

    async def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt and return the completion text."""
        ...

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        """Send a system + user prompt with a GBNF grammar constraint."""
        ...


class ClaudeLLMClient:
    """LLM client backed by Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        # Anthropic API path here has no grammar hook; fallback to normal completion.
        return await self.complete(system, user)


class CodexLLMClient:
    """LLM client backed by OpenAI's API (Codex-compatible models)."""

    def __init__(
        self,
        api_key: str,
        model: str = "codex-mini-latest",
        max_tokens: int = 4096,
        base_url: str = "",
    ) -> None:
        from openai import AsyncOpenAI

        if base_url:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, dict):
                    item_text = item.get("text")
                    if isinstance(item_text, str):
                        parts.append(item_text)
            return "".join(parts)
        return ""

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        # Generic OpenAI/Codex path may not support GBNF directly; fallback.
        return await self.complete(system, user)


class LlamaCppLLMClient:
    """LLM client backed by a local llama.cpp server (OpenAI-compatible API)."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 4096,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "local",
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            # llama.cpp accepts grammar via extra_body on compatible endpoints.
            extra_body={"grammar": grammar},
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""


def create_llm_client(
    *,
    provider: str,
    model: str,
    max_tokens: int,
    anthropic_api_key: str = "",
    openai_api_key: str = "",
    openai_base_url: str = "",
    llama_cpp_base_url: str = "",
    llama_cpp_api_key: str = "",
) -> LLMClient:
    """Construct an LLM client for the requested provider."""
    normalized = provider.lower().strip()

    if normalized == "anthropic":
        if not anthropic_api_key:
            raise ValueError("AGEOM_ANTHROPIC_API_KEY not set")
        return ClaudeLLMClient(
            api_key=anthropic_api_key,
            model=model,
            max_tokens=max_tokens,
        )

    if normalized == "codex":
        if not openai_api_key:
            raise ValueError("AGEOM_OPENAI_API_KEY not set")
        return CodexLLMClient(
            api_key=openai_api_key,
            model=model,
            max_tokens=max_tokens,
            base_url=openai_base_url,
        )

    if normalized in {"llama_cpp", "local"}:
        return LlamaCppLLMClient(
            model=model,
            max_tokens=max_tokens,
            base_url=llama_cpp_base_url or "http://127.0.0.1:8080/v1",
            api_key=llama_cpp_api_key or "local",
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
