#!/usr/bin/env python3
"""Smoke-test a local llama.cpp OpenAI-compatible server.

Checks:
1) Server is reachable and returns /models.
2) Model can follow a strict JSON output contract.
3) Model returns sensible outputs for simple arithmetic and factual prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _load_dotenv_value(key: str, dotenv_path: Path) -> str:
    env_val = os.getenv(key)
    if env_val is not None and env_val != "":
        return env_val
    if not dotenv_path.exists():
        return ""
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return ""


def _request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} at {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed request to {url}: {exc}") from exc


def _extract_chat_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response shape: {response}") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content).strip()


def _chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> str:
    response = _request_json(
        method="POST",
        url=f"{base_url.rstrip('/')}/chat/completions",
        payload={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 96,
        },
        api_key=api_key,
        timeout_s=timeout_s,
    )
    return _extract_chat_text(response)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return t


def main() -> int:
    dotenv_path = Path(".env")

    parser = argparse.ArgumentParser(
        description="Test local llama.cpp server health and output quality."
    )
    parser.add_argument(
        "--base-url",
        default=_load_dotenv_value("AGEOM_LLAMA_CPP_BASE_URL", dotenv_path)
        or "http://127.0.0.1:18080/v1",
        help="OpenAI-compatible base URL (default: AGEOM_LLAMA_CPP_BASE_URL or http://127.0.0.1:18080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=_load_dotenv_value("AGEOM_LLAMA_CPP_API_KEY", dotenv_path) or "local",
        help="API key (default: AGEOM_LLAMA_CPP_API_KEY or local)",
    )
    parser.add_argument(
        "--model",
        default=_load_dotenv_value("AGEOM_HUNTER_LLM_MODEL", dotenv_path),
        help="Model ID to test (default: AGEOM_HUNTER_LLM_MODEL; falls back to first server model)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout per request in seconds (default: 20)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    api_key = args.api_key
    timeout_s = float(args.timeout)

    print(f"[info] base_url={base_url}")

    # 1) Health + model list
    try:
        models_payload = _request_json(
            method="GET",
            url=f"{base_url}/models",
            payload=None,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    except RuntimeError as exc:
        print(f"[fail] could not reach llama server at {base_url}: {exc}")
        print("[hint] Start the server first (example: scripts/run_llama_8b_server.sh)")
        return 1
    model_entries = models_payload.get("data", [])
    model_ids = [m.get("id", "") for m in model_entries if isinstance(m, dict)]
    if not model_ids:
        print("[fail] /models returned no model IDs")
        return 1
    print(
        f"[pass] /models returned {len(model_ids)} model(s): {', '.join(model_ids[:3])}"
    )

    model = args.model or model_ids[0]
    if model not in model_ids:
        print(
            f"[warn] requested model '{model}' not in /models; using '{model_ids[0]}'"
        )
        model = model_ids[0]
    print(f"[info] using model={model}")

    failures: list[str] = []

    # 2) JSON contract check
    json_prompt = (
        "Return ONLY valid JSON (no markdown) exactly like this schema: "
        '{"status":"ok","value":7}'
    )
    try:
        json_out = _chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict output-format assistant.",
                },
                {"role": "user", "content": json_prompt},
            ],
            timeout_s=timeout_s,
        )
        try:
            parsed = json.loads(_strip_fences(json_out))
            if not (
                isinstance(parsed, dict)
                and parsed.get("status") == "ok"
                and parsed.get("value") == 7
            ):
                failures.append(f"json_contract_bad_values: {json_out}")
            else:
                print("[pass] JSON contract output is valid and correct")
        except json.JSONDecodeError:
            failures.append(f"json_contract_not_json: {json_out}")
    except RuntimeError as exc:
        failures.append(f"json_contract_request_failed: {exc}")

    # 3) Arithmetic sanity
    try:
        math_out = _chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer only with the final integer result.",
                },
                {"role": "user", "content": "19 + 23 = ?"},
            ],
            timeout_s=timeout_s,
        )
        m = re.search(r"-?\d+", math_out)
        if not m or int(m.group(0)) != 42:
            failures.append(f"math_incorrect: {math_out}")
        else:
            print("[pass] arithmetic sanity check (19 + 23 = 42)")
    except RuntimeError as exc:
        failures.append(f"math_request_failed: {exc}")

    # 4) Simple factual sanity
    try:
        fact_out = _chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": "Answer in one word only."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            timeout_s=timeout_s,
        )
        if "paris" not in fact_out.lower():
            failures.append(f"fact_incorrect: {fact_out}")
        else:
            print("[pass] factual sanity check (capital of France)")
    except RuntimeError as exc:
        failures.append(f"fact_request_failed: {exc}")

    if failures:
        print("[fail] llama server test failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[pass] llama server health + output sanity checks passed")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    try:
        code = main()
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"[fail] unexpected error: {exc}")
        code = 1
    print(f"[info] completed in {time.time() - t0:.1f}s")
    raise SystemExit(code)
