# ML Intern — Execution Sandbox Analysis

## Training Job Execution — Three Tiers

### 1. HF Jobs (remote, primary for cloud training)
**File:** `agent/tools/jobs_tool.py`

- Runs on remote HuggingFace infrastructure (Docker containers)
- Selectable hardware: CPU (cpu-basic, cpu-upgrade) and GPU (T4 through H100x8)
- Ephemeral storage — models must `push_to_hub=True` or they're lost
- Default image: `ghcr.io/astral-sh/uv:python3.12-bookworm`
- Dependencies auto-managed via `uv`
- Log streaming with retry logic for multi-hour jobs
- Job tracking via `session._running_job_ids` for cancellation

### 2. Sandbox (remote, persistent dev environment)
**File:** `agent/tools/sandbox_tool.py`, `agent/tools/sandbox_client.py`

- Persistent remote Linux environment on HF Spaces
- Created by duplicating template Space (`burtenshaw/sandbox`)
- Survives across tool calls within a session
- Good for iterative testing before submitting full jobs
- Can be CPU or GPU-based (t4-small minimum for CUDA)

### 3. Local Mode (CLI, runs on user's machine)
**File:** `agent/tools/local_tools.py`

- Used when `local_mode=True` in ToolRouter initialization
- Bash via `subprocess.run(shell=True)`
- Max timeout: 36,000 seconds (10 hours) for training
- File tools: read, write, edit with read-before-write enforcement
- Atomic writes via temp file + `os.replace()`

## Approval / Safety Gates

| Action | Approval Required? |
|--------|-------------------|
| GPU jobs (HF) | Always |
| CPU jobs (HF) | If `confirm_cpu_jobs=True` (default) |
| File uploads/deletes | Always |
| Sandbox creation | Always |
| All actions | Skipped if `yolo_mode=True` or headless mode |

Users can edit training scripts in the approval dialog before execution.
On interrupt, all running jobs are cancelled via `HfApi.cancel_job()`.

## Configuration Controls

```python
class Config:
    yolo_mode: bool = False          # Auto-approve all tools
    confirm_cpu_jobs: bool = True    # Require approval for CPU jobs
    auto_file_upload: bool = False   # Auto-approve file uploads
    max_iterations: int = 300        # Prevent infinite agent loops
```

## Key Files

| File | Purpose |
|------|---------|
| `agent/tools/jobs_tool.py` | HF Jobs API: submit, log, cancel training jobs |
| `agent/tools/sandbox_tool.py` | Create/manage persistent HF Space sandboxes |
| `agent/tools/sandbox_client.py` | Low-level sandbox client (HTTPS to Space) |
| `agent/tools/local_tools.py` | Local execution fallback (bash, read, write, edit) |
| `agent/core/agent_loop.py` | Agentic loop, tool execution, approval flow |
| `agent/core/tools.py` | ToolRouter: registers built-in + MCP tools |
| `agent/core/session.py` | Session state (context, config, job IDs, events) |
| `agent/config.py` | Approval and runtime config |

## Local LLM + Local Training Workflow (Current Gap)

When using a local Ollama model in local mode, the agent executes bash commands
directly on the user's machine. There is currently no dedicated abstraction for:

1. **Local Docker-based training** — The agent has `hf_jobs` for remote HF infra
   but no equivalent for launching local Docker containers (e.g., unsloth/unsloth)
   with GPU passthrough.

2. **Sandbox folder convention** — No enforcement that generated files go to a
   specific local directory (e.g., `sandbox/`).

3. **Local-to-Hub upload flow** — The agent can upload via `hf_repo_files` but
   there's no integrated workflow that chains: local training → checkpoint save
   → hub upload.

The agent falls back to raw `bash` commands for Docker execution, which works
but requires the LLM to correctly compose `docker run` commands with volume
mounts, GPU flags, and environment variables — error-prone for smaller models.

---

## Problem: Current Local Docker Training Workflow Fails

### Observed Behavior

When running a prompt like:
```
ml-intern --model ollama/gemma4:e4b "For the SAM model example in sandbox/sa_peft_tutorial,
do the full model training locally on the 1st GPU using unsloth/unsloth:latest docker with
--gpus device=0, keep all files under sandbox/" --max-iterations=100
```

The agent:
1. Correctly reads the sandbox directory and existing files
2. Correctly writes training scripts to `sandbox/`
3. **Fails** at Docker execution — "simulates" instead of running, or composes
   incorrect `docker run` commands (missing mounts, wrong GPU flags)
4. Never reaches the HF Hub upload step

### Root Cause

- The **system prompt** (`agent/prompts/system_prompt_v3.yaml`) has no guidance
  on local Docker + GPU training workflows
- The **4B model** (gemma4:e4b) struggles to compose complex multi-flag Docker
  CLI commands from first principles
- There is no **dedicated tool** for local Docker training — the agent must
  use raw `bash`, which is error-prone
- The `hf_jobs` tool only targets remote HF infrastructure, not local Docker

### What Currently Works

| Capability | Status | File |
|-----------|--------|------|
| Local bash execution | Works | `agent/tools/local_tools.py` |
| Docker commands via bash | Works (fragile) | via bash tool |
| File I/O in sandbox/ | Works | local_tools read/write/edit |
| Upload to HF Hub | Works | `agent/tools/hf_repo_files_tool.py` |
| Long timeouts (10h) | Works | `local_tools.py` MAX_TIMEOUT=36000 |
| HF token in env | Works | `agent/core/session.py` |
| Example training scripts | Exist | `sandbox/run_local_peft_training.py` |

### What's Missing

1. **No system prompt guidance** for local Docker training patterns
2. **No helper tool** to build `docker run` commands with:
   - `--gpus device=0` (GPU passthrough)
   - `-v $(pwd)/sandbox/<project>:/workspace` (volume mounts)
   - `-e HF_TOKEN=$HF_TOKEN` (env vars)
   - `--rm` (cleanup after completion)
3. **No documented pattern** for: train in Docker → save to sandbox/ → upload to Hub
4. **No integration** between Docker container output and HF upload in agent prompts

---

## Proposed Fix: Three Options

### Option A — System Prompt Update (lightest touch)

Add a "Local Docker Training" section to `agent/prompts/system_prompt_v3.yaml`
with the exact `docker run` pattern, sandbox folder conventions, and upload flow.
The LLM gets clear instructions but still has to compose commands correctly.

**Pros:** Minimal code changes, works immediately
**Cons:** Still relies on LLM to compose Docker CLI correctly; fragile with small models

### Option B — Dedicated `local_training` Tool (medium effort)

A new tool in `agent/tools/` that wraps the Docker workflow. Takes a script path,
Docker image, GPU device, and sandbox dir — then builds/runs the `docker run`
command, monitors it, and returns logs. Removes the Docker CLI composition burden
from the LLM.

**Pros:** Reliable, model-agnostic, handles edge cases
**Cons:** More code to maintain, new tool for LLM to learn

### Option C — Both (recommended)

System prompt guidance for overall workflow planning + a `local_training` tool
that handles Docker execution mechanics. The LLM plans the workflow and writes
scripts; the tool handles the messy Docker mechanics.

**Pros:** Best of both — LLM understands the workflow AND has a reliable tool
**Cons:** Most implementation effort

### Proposed Workflow (Option C)

```
1. Agent writes training script → sandbox/<project>/train.py
2. Agent calls local_training tool with:
   - script: sandbox/<project>/train.py
   - image: unsloth/unsloth:latest
   - gpu: 0
   - sandbox_dir: sandbox/<project>
3. Tool executes:
   docker run --rm \
     --gpus device=0 \
     -v $(pwd)/sandbox/<project>:/workspace \
     -e HF_TOKEN=$HF_TOKEN \
     unsloth/unsloth:latest \
     python /workspace/train.py
4. Tool streams logs back, waits for completion
5. Agent verifies outputs in sandbox/<project>/output/
6. Agent uploads sandbox/<project>/output/ to HF Hub via hf_repo_files
```

### Implementation Estimate

| Component | Files to Change | Effort |
|-----------|----------------|--------|
| System prompt update | `agent/prompts/system_prompt_v3.yaml` | Small |
| `local_training` tool | New `agent/tools/local_training_tool.py` | Medium |
| Tool registration | `agent/core/tools.py` (add to local_mode tools) | Small |
| Tool spec for LLM | Within the new tool file | Small |
| Config: default sandbox dir | `agent/config.py` | Small |
