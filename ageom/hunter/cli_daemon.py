"""Persistent socket daemon for CLI-backed shim providers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ageom.hunter.shim_pool import ShimPoolClient


def _jsonrpc_result(request_id: object, result: dict[str, object]) -> bytes:
    return (json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n").encode()


def _jsonrpc_error(request_id: object, code: int, message: str) -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    ).encode()


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    client: ShimPoolClient | None,
    cli: str,
    model: str,
    max_tokens: int,
    fake_mode: bool,
) -> None:
    request_count = 0
    first_completion = True
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode())
            except Exception as exc:
                writer.write(_jsonrpc_error(None, -32700, f"Parse error: {exc}"))
                await writer.drain()
                continue

            request_id = message.get("id")
            method = message.get("method")
            params = message.get("params") or {}

            try:
                if method == "ping":
                    writer.write(
                        _jsonrpc_result(
                            request_id,
                            {
                                "ok": True,
                                "pid": os.getpid(),
                                "cli": cli,
                                "model": model,
                                "requestCount": request_count,
                            },
                        )
                    )
                    await writer.drain()
                    continue

                if method not in {"complete", "complete_with_grammar"}:
                    writer.write(_jsonrpc_error(request_id, -32601, f"Unknown method: {method}"))
                    await writer.drain()
                    continue

                request_count += 1
                system = str(params.get("system") or "")
                user = str(params.get("user") or "")
                grammar = str(params.get("grammar") or "")

                if fake_mode:
                    text = (
                        f"fake cli={cli} pid={os.getpid()} count={request_count} "
                        f"model={model} system={system} user={user} grammar={bool(grammar)}"
                    )
                else:
                    assert client is not None
                    if method == "complete_with_grammar":
                        text = await client.complete_with_grammar(system, user, grammar)
                    else:
                        text = await client.complete(system, user)

                writer.write(
                    _jsonrpc_result(
                        request_id,
                        {
                            "text": text,
                            "pid": os.getpid(),
                            "requestCount": request_count,
                            "model": model,
                            "cli": cli,
                            "coldStart": first_completion,
                        },
                    )
                )
                await writer.drain()
                first_completion = False
            except Exception as exc:
                writer.write(_jsonrpc_error(request_id, -32000, str(exc)))
                await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cli", required=True, choices=["claude", "codex"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--use-agent-layer", action="store_true")
    args = parser.parse_args()

    socket_path = Path(args.socket)
    try:
        socket_path.unlink(missing_ok=True)
    except Exception:
        pass

    fake_mode = os.getenv("AGEOM_CLI_SHIM_DAEMON_FAKE", "").strip() == "1"
    client: ShimPoolClient | None = None
    if not fake_mode:
        client = ShimPoolClient(
            cli=args.cli,
            model=args.model,
            max_tokens=args.max_tokens,
            use_agent_layer=args.use_agent_layer,
        )

    server = await asyncio.start_unix_server(
        lambda reader, writer: _handle_connection(
            reader,
            writer,
            client=client,
            cli=args.cli,
            model=args.model,
            max_tokens=args.max_tokens,
            fake_mode=fake_mode,
        ),
        path=str(socket_path),
    )
    try:
        async with server:
            await server.serve_forever()
    finally:
        if client is not None:
            await client.close()
        try:
            socket_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(_main())
