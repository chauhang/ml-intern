# Plan: Add Ollama Support for Local Model Inference

## Context

The agent currently supports Anthropic and HuggingFace Router models via litellm. We added support for local Ollama models so the agent can run with `ollama/qwen3-coder-q3:latest`.

**Key design decision:** Ollama's native `/api/chat` endpoint (litellm's `ollama/` provider) doesn't properly structure tool call responses — the model outputs tool call JSON as plain text content. The fix is to route through Ollama's **OpenAI-compatible `/v1` endpoint** by rewriting `ollama/<model>` to `openai/<model>` with `api_base=localhost:11434/v1`.

## Changes (7 files)

### 1. `agent/core/agent_loop.py` — Core resolver function

Renamed `_resolve_hf_router_params` → `_resolve_llm_params`. Added Ollama block that rewrites `ollama/<model>` to `openai/<model>` targeting Ollama's `/v1` endpoint with a dummy API key.

### 2. `backend/routes/agent.py` — Import rename

Updated import and call sites to match the renamed function.

### 3. `agent/tools/research_tool.py` — Local resolver copy

Added the same Ollama rewrite block to the file's own copy of the resolver.

### 4. `agent/context_manager/manager.py` — Compact function

Added Ollama model/api_base/api_key rewriting in the compact kwargs.

### 5. `agent/core/session.py` — Max tokens map

Added `"ollama/qwen3-coder-q3:latest": 32_768` to `_MAX_TOKENS_MAP`.

### 6. `agent/main.py` — UI + startup logic

- Added `ollama/qwen3-coder-q3:latest` to `SUGGESTED_MODELS`
- Expanded `_is_valid_model_id` to accept `ollama/` prefix
- Made `_handle_slash_command` async, added Ollama readiness check on `/model` switch
- Skip HF token requirement for local Ollama models
- Added Ollama readiness checks in both interactive and headless startup

### 7. `agent/utils/ollama_utils.py` — New helper module

Utility functions: `is_ollama_running()`, `is_model_available()`, `pull_ollama_model()`, `ensure_ollama_readiness()`.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_API_BASE` | Override Ollama server address (optional) | `http://localhost:11434` |

## Usage

```bash
# Headless
ml-intern --model ollama/qwen3-coder-q3:latest "What is 2+2?"

# Interactive — switch at runtime
/model ollama/qwen3-coder-q3:latest
```
