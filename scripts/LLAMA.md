# LLAMA Server Notes

This file documents what was done to get local llama serving and smoke tests passing in this repo.

## What Was Done

1. Verified model and env wiring:
   - Model exists at `models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.
   - `.env` points to `AGEOM_LLAMA_CPP_BASE_URL=http://127.0.0.1:18080/v1`.

2. Ran the smoke test:
   - `python3 scripts/test_llama_server.py`
   - Initial result: connection refused because the server was not successfully starting.

3. Debugged Python server startup (`python -m llama_cpp.server`):
   - Reproduced model load failure in the Python binding.
   - Root cause observed from runtime logs: Metal backend init failure (`failed to create command queue` / `failed to initialize Metal backend`), preventing context creation.

4. Switched to native `llama.cpp` server path:
   - Built native server:
     - `git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp`
     - `cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp -DLLAMA_METAL=OFF -DGGML_METAL=OFF`
     - `cmake --build /tmp/llama.cpp/build -j 4 --target llama-server`
   - Started with CPU-only flags:
     - `/tmp/llama.cpp/build/bin/llama-server --model models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 18080 --api-key local --ctx-size 512 --device none --fit off --n-gpu-layers 0 --verbosity 0`

5. Re-ran smoke test and confirmed pass:
   - `python3 scripts/test_llama_server.py`
   - Checks passed: `/models`, JSON contract, arithmetic sanity, factual sanity.

## Run Script Update

`scripts/run_llama_8b_server.sh` now:

- Prefers native `llama-server` in `auto` mode.
- Supports explicit modes:
  - `LLAMA_SERVER_MODE=auto` (default)
  - `LLAMA_SERVER_MODE=native`
  - `LLAMA_SERVER_MODE=python`
- Detects binary via:
  1. `LLAMA_SERVER_BIN` (if set)
  2. `llama-server` on `PATH`
  3. `/tmp/llama.cpp/build/bin/llama-server`
- Uses CPU-safe defaults for native mode:
  - `LLAMA_DEVICE=none`
  - `LLAMA_GPU_LAYERS=0`
  - `LLAMA_FIT=off`
  - `LLAMA_VERBOSITY=0`
- Falls back to `python -m llama_cpp.server` only if native is unavailable (or `LLAMA_SERVER_MODE=python`).

## Quick Start

```bash
scripts/run_llama_8b_server.sh
python3 scripts/test_llama_server.py
```

## TODO

- TODO: Determine a reliable GPU-enabled build/runtime path on this machine (Metal-capable compile + runtime flags) so we can run `llama-server` with GPU acceleration instead of CPU-only mode.
