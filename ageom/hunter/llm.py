"""LLM client abstraction for the Hunter agent."""

from __future__ import annotations

import asyncio
import functools
import json as _json
import os
import random
import time
from pathlib import Path
from asyncio.subprocess import PIPE
from typing import Protocol, runtime_checkable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


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


class SubprocessCLIClient:
    """LLM client that shells out to a CLI tool (claude, codex, gemini)."""

    _CLI_VARIANTS = {"claude", "codex", "gemini"}
    _CODEX_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "codex")
    _NON_CODEX_MODEL_PREFIXES = ("claude", "gemini", "llama", "qwen", "deepseek")

    def __init__(
        self,
        cli: str,
        model: str,
        max_tokens: int,
        *,
        use_agent_layer: bool = False,
    ) -> None:
        if cli not in self._CLI_VARIANTS:
            raise ValueError(f"Unknown CLI variant: {cli!r}")
        self._cli = cli
        self._model = model
        self._max_tokens = max_tokens
        self._use_agent_layer = use_agent_layer
        self._timeout_s = _env_float("AGEOM_SUBPROCESS_TIMEOUT_S", 90.0)
        self._max_retries = _env_int("AGEOM_SUBPROCESS_MAX_RETRIES", 1, min_value=0)
        self._retry_backoff_s = _env_float("AGEOM_SUBPROCESS_RETRY_BACKOFF_S", 1.5)

    def _codex_model_arg(self) -> str:
        """Return a codex-compatible model name, or empty when model should be omitted."""
        model = self._model.strip()
        if not model:
            return ""
        lowered = model.lower()
        if lowered.startswith(self._NON_CODEX_MODEL_PREFIXES):
            return ""
        if lowered.startswith(self._CODEX_MODEL_PREFIXES):
            return model
        return ""

    def _build_cmd(self, system: str) -> list[str]:
        """Build the argv for the subprocess."""
        cmd: list[str] = []
        if self._use_agent_layer:
            cmd.append("al")

        if self._cli == "claude":
            cmd += ["claude", "-p", "--output-format", "text"]
            if system:
                cmd += ["--system-prompt", system]
            if self._model:
                cmd += ["--model", self._model]
        elif self._cli == "codex":
            # Rely on JSONL events from stdout; avoid output-file writes that can fail in sandboxes.
            cmd += ["codex", "exec", "--json", "--ephemeral"]
            for override in _codex_mcp_disable_overrides():
                cmd += ["-c", override]
            model = self._codex_model_arg()
            if model:
                cmd += ["-m", model]
        elif self._cli == "gemini":
            cmd += ["gemini", "-p", "--output-format", "text", "--extensions", ""]
            if self._model:
                cmd += ["--model", self._model]
        return cmd

    def _build_stdin(self, system: str, user: str) -> str:
        """Build the text piped to stdin."""
        if self._cli == "claude":
            # System prompt is passed via --system-prompt flag
            return user
        # codex / gemini: prepend system to user prompt
        if system:
            return f"[System]\n{system}\n\n{user}"
        return user

    def _parse_output(self, raw: str) -> str:
        """Extract the assistant reply from stdout."""
        if self._cli == "codex":
            # codex --json emits JSONL events; pick last assistant message
            last_text = ""
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if event.get("role") == "assistant":
                    last_text = event.get("content", "")
                elif isinstance(event.get("message"), str):
                    last_text = event["message"]
                else:
                    item = event.get("item")
                    if isinstance(item, dict):
                        item_type = str(item.get("type", "")).strip().lower()
                        if item_type in {"agent_message", "assistant_message"}:
                            if isinstance(item.get("text"), str):
                                last_text = item["text"]
                            elif isinstance(item.get("content"), str):
                                last_text = item["content"]
                            elif isinstance(item.get("content"), list):
                                parts: list[str] = []
                                for chunk in item["content"]:
                                    if isinstance(chunk, str):
                                        parts.append(chunk)
                                    elif isinstance(chunk, dict):
                                        text = chunk.get("text")
                                        if isinstance(text, str):
                                            parts.append(text)
                                if parts:
                                    last_text = "".join(parts)
            return last_text or raw.strip()
        
        # gemini / claude: raw stdout is the reply, but may have log preamble
        lines = raw.strip().splitlines()
        content_lines = []
        skip_prefixes = (
            "Loaded cached credentials",
            "Loading extension",
            "MCP server",
            "Error when talking to Gemini API",
            "Full report available at",
        )
        for line in lines:
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            content_lines.append(line)
        
        return "\n".join(content_lines).strip()

    async def complete(self, system: str, user: str) -> str:
        cmd = self._build_cmd(system)
        stdin_text = self._build_stdin(system, user)
        # Strip CLAUDECODE env var so nested Claude CLI sessions don't refuse.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        attempts = max(1, self._max_retries + 1)
        last_error = ""

        for attempt in range(1, attempts + 1):
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env,
            )
            started = time.time()
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_text.encode()),
                    timeout=self._timeout_s,
                )
            except asyncio.TimeoutError as exc:
                elapsed_ms = (time.time() - started) * 1000.0
                await self._terminate_process(proc)
                self._log_subprocess_event(
                    "PROMPT_SUBPROCESS_TIMEOUT",
                    {
                        "cli": self._cli,
                        "model": self._model,
                        "attempt": attempt,
                        "attempts_total": attempts,
                        "timeout_s": self._timeout_s,
                        "elapsed_ms": round(elapsed_ms, 2),
                    },
                )
                last_error = (
                    f"{self._cli} CLI timed out after {self._timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts})"
                )
                if attempt >= attempts:
                    raise RuntimeError(last_error) from exc
                await self._sleep_before_retry(attempt)
                continue

            stderr_text = stderr.decode().strip()
            if proc.returncode != 0:
                last_error = (
                    f"{self._cli} CLI exited with code {proc.returncode}: {stderr_text}"
                )
                if attempt < attempts and self._is_transient_error(stderr_text):
                    self._log_subprocess_event(
                        "PROMPT_SUBPROCESS_RETRY",
                        {
                            "cli": self._cli,
                            "model": self._model,
                            "attempt": attempt,
                            "attempts_total": attempts,
                            "reason": stderr_text[:200],
                        },
                    )
                    await self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(last_error)

            return self._parse_output(stdout.decode())

        raise RuntimeError(last_error or f"{self._cli} CLI failed without output")

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        # CLI tools have no GBNF support; fall back to plain completion.
        return await self.complete(system, user)

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        waiter = getattr(proc, "wait", None)
        if callable(waiter):
            try:
                await asyncio.wait_for(waiter(), timeout=2.0)
            except Exception:
                return

    async def _sleep_before_retry(self, attempt: int) -> None:
        base = max(0.0, self._retry_backoff_s)
        jitter = random.uniform(0.0, 0.25)
        await asyncio.sleep(base * attempt + jitter)

    def _is_transient_error(self, text: str) -> bool:
        lower = text.lower()
        markers = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "timeout",
            "timed out",
            "rate limit",
            "temporarily unavailable",
            "connection reset",
            "network error",
            "econnreset",
            "socket hang up",
            "http error",
        )
        return any(m in lower for m in markers)

    def _log_subprocess_event(self, event_type: str, payload: dict[str, object]) -> None:
        try:
            from ageom.telemetry import get_current_stage, log_event

            stage = get_current_stage() or "prompt_dispatch"
            log_event(
                "llm",
                phase=stage,
                event_type=event_type,
                stage=stage,
                provider=self._cli,
                model=self._model,
                payload=payload,
            )
        except Exception:
            return


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@functools.lru_cache(maxsize=1)
def _codex_mcp_disable_overrides() -> tuple[str, ...]:
    """Return codex config overrides that disable MCP servers for shim calls.

    This avoids fragile MCP transport failures in non-interactive `codex exec`
    paths and reduces startup overhead for short prompt dispatches.
    """
    if not _env_bool("AGEOM_CODEX_SHIM_DISABLE_MCP", True):
        return ()

    config_path = Path(
        os.getenv("AGEOM_CODEX_CONFIG_PATH", "~/.codex/config.toml")
    ).expanduser()
    if not config_path.exists():
        return ()

    try:
        data = tomllib.loads(config_path.read_text())
    except Exception:
        return ()

    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return ()

    overrides: list[str] = []
    for name in servers:
        if isinstance(name, str) and name.strip():
            overrides.append(f"mcp_servers.{name}.enabled=false")
    return tuple(overrides)


_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "claude_cli": "sonnet",
    "claude_shim": "sonnet",
    "codex": "gpt-5.3-codex",
    "codex_cli": "gpt-5.3-codex",
    "codex_shim": "gpt-5.3-codex",
    "gemini_cli": "gemini-2.5-pro",
    "gemini_shim": "gemini-2.5-pro",
    "llama_cpp": "qwen2.5-coder:7b",
    "local": "qwen2.5-coder:7b",
}
_CODEX_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "codex")
_CLAUDE_MODEL_PREFIXES = ("claude",)
_CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku"}
_GEMINI_MODEL_PREFIXES = ("gemini",)
_GEMINI_MODEL_ALIASES = {"pro", "flash", "flash-lite"}


def _looks_like_codex_model(model: str) -> bool:
    lowered = model.lower().strip()
    return lowered.startswith(_CODEX_MODEL_PREFIXES)


def _looks_like_claude_model(model: str) -> bool:
    lowered = model.lower().strip()
    return lowered.startswith(_CLAUDE_MODEL_PREFIXES) or lowered in _CLAUDE_MODEL_ALIASES


def _looks_like_gemini_model(model: str) -> bool:
    lowered = model.lower().strip()
    return lowered.startswith(_GEMINI_MODEL_PREFIXES) or lowered in _GEMINI_MODEL_ALIASES


def _normalized_model_for_provider(provider: str, model: str) -> str:
    """Pick a provider-compatible model and validate obvious misconfigurations."""
    cleaned = model.strip()
    default_model = _DEFAULT_MODELS.get(provider, "").strip()

    if provider in {"llama_cpp", "local"}:
        return cleaned or default_model

    if provider in {"codex", "codex_cli", "codex_shim"}:
        if not cleaned:
            return default_model
        if _looks_like_codex_model(cleaned):
            return cleaned
        if _looks_like_claude_model(cleaned) or _looks_like_gemini_model(cleaned):
            return default_model
        raise ValueError(
            f"Model '{model}' is not codex-compatible for provider '{provider}'. "
            f"Use a codex/openai model such as '{default_model}'."
        )

    if provider in {"anthropic", "claude_cli", "claude_shim"}:
        if not cleaned:
            return default_model
        if _looks_like_claude_model(cleaned):
            return cleaned
        if _looks_like_codex_model(cleaned) or _looks_like_gemini_model(cleaned):
            return default_model
        if provider == "claude_cli":
            # Claude CLI accepts additional aliases/custom model names.
            return cleaned
        raise ValueError(
            f"Model '{model}' is not claude-compatible for provider '{provider}'."
        )

    if provider in {"gemini_cli", "gemini_shim"}:
        if not cleaned:
            return default_model
        if _looks_like_gemini_model(cleaned):
            return cleaned
        if _looks_like_codex_model(cleaned) or _looks_like_claude_model(cleaned):
            return default_model
        # Gemini CLI also accepts provider-side aliases.
        return cleaned

    return cleaned or default_model


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
    use_agent_layer: bool = False,
) -> LLMClient:
    """Construct an LLM client for the requested provider."""
    normalized = provider.lower().strip()
    resolved_model = _normalized_model_for_provider(normalized, model)

    if normalized == "anthropic":
        if not anthropic_api_key:
            raise ValueError("AGEOM_ANTHROPIC_API_KEY not set")
        return ClaudeLLMClient(
            api_key=anthropic_api_key,
            model=resolved_model,
            max_tokens=max_tokens,
        )

    if normalized == "codex":
        if not openai_api_key:
            raise ValueError("AGEOM_OPENAI_API_KEY not set")
        return CodexLLMClient(
            api_key=openai_api_key,
            model=resolved_model,
            max_tokens=max_tokens,
            base_url=openai_base_url,
        )

    if normalized in {"llama_cpp", "local"}:
        return LlamaCppLLMClient(
            model=resolved_model,
            max_tokens=max_tokens,
            base_url=llama_cpp_base_url or "http://127.0.0.1:8080/v1",
            api_key=llama_cpp_api_key or "local",
        )

    if normalized == "claude_cli":
        return SubprocessCLIClient(
            cli="claude", model=resolved_model, max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )

    if normalized == "codex_cli":
        return SubprocessCLIClient(
            cli="codex", model=resolved_model, max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )

    if normalized == "gemini_cli":
        return SubprocessCLIClient(
            cli="gemini", model=resolved_model, max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )

    if normalized == "claude_shim":
        from ageom.hunter.cli_socket_shim import CLISocketShimClient

        return CLISocketShimClient(
            cli="claude", model=resolved_model, max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )

    if normalized == "codex_shim":
        from ageom.hunter.cli_socket_shim import CLISocketShimClient

        return CLISocketShimClient(
            cli="codex", model=resolved_model, max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )

    if normalized == "gemini_shim":
        from ageom.hunter.gemini_shim import GeminiShimClient

        return GeminiShimClient(
            model=resolved_model,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
