# `agy-token-viz` (`agy --visualize-tokens` & `jetski --visualize-tokens`)

A real-time CLI token-usage visualizer for **Antigravity CLI (`agy`)** and **Jetski CLI (`jetski`)**. Passing `--visualize-tokens` pins a 2-line `tqdm`-style status bar to the bottom of your terminal while `agy` or `jetski` runs normally above it (both inline and full-session alt-screen TUIs), attributing net context tokens across 6 activity categories using tool-signature guardrails + a local GGUF model (`llama-cpp-python`).

```text
Tokens 70.8k │████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▫▫│ In:56.8k Cache:328k Out:14.1k Think:8.1k [9 steps | qwen2.5-3b]
█ Plan 8.1k (11%)  ▓ Explore 57.1k (81%)  ▒ Code 2.4k (3%)  ░ Cmd 0 (0%)  ▪ Test 143 (0%)  ▫ Comm 3.0k (4%)
```

When `agy` or `jetski` exits, the footer is cleared and a per-category session summary (including both net context `In` and raw API `Billed` uncached input) is printed to `stderr`.

---

## Setup

Requires **Python 3.10+**, **macOS or Linux** (POSIX PTY), and an installed `agy` and/or `jetski` binary.

```bash
# 1. Create the repo-local virtualenv expected by `bin/agy` and `bin/jetski` (or set TOKEN_VIZ_PYTHON / AGY_VIZ_PYTHON)
python3 -m venv .venv
.venv/bin/pip install -e .

# 2. (Optional) Pre-download and verify default GGUF weights (Qwen2.5-3B-Instruct Q4_K_M, ~2.0 GB)
#    Weights are shared across `agy` and `jetski`. Export HF_TOKEN="hf_..." first if pulling gated models.
./scripts/preload_model.sh

# 3. Prepend `bin/` (`bin/agy` and `bin/jetski`) to PATH in ~/.zshrc (use --dry-run to preview, or --rc-file ~/.bashrc)
./scripts/install_shim.sh
source ~/.zshrc
```

---

## Usage

Once `bin/` is ahead of the real `agy` / `jetski` binaries on `PATH`, normal invocations pass straight through untouched; adding `--visualize-tokens` (or any `--viz-*` flag) enables visualization:

```bash
# Normal agy / jetski (zero Python overhead; execs underlying binary directly)
agy -p "Quick question"
jetski -p "Quick question"

# Interactive TUI session with live 2-line bottom bar (supports both inline and alt-screen TUIs)
agy --visualize-tokens
jetski --visualize-tokens

# Single-shot / piped run (stdout stays clean; 1-line bar & summary go to stderr)
agy --visualize-tokens -p "Summarize this repo" > summary.md
jetski --visualize-tokens -p "Summarize this repo" > summary.md

# Skip local LLM loading and classify turns via deterministic rules only
agy --visualize-tokens --viz-heuristic-only
jetski --visualize-tokens --viz-heuristic-only

# Select a model preset (`qwen2.5-3b`, `qwen2.5-7b`, `llama-3.2-3b`, `llama-3.1-8b`, ...),
# Hugging Face `<org>/<repo>:<file.gguf>`, or local `.gguf` file
jetski --visualize-tokens --viz-model qwen2.5-7b
agy --visualize-tokens --viz-model "bartowski/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf"
agy --visualize-tokens --viz-model ~/models/custom-q4_k_m.gguf

# Custom activity categories
jetski --visualize-tokens --viz-categories "Planning,Research,Coding,Commands,Testing,Review"

# Include prior turns when resuming a session
agy --visualize-tokens --viz-include-history --continue
jetski --visualize-tokens --viz-include-history --continue
```

Direct invocation without the shell shims:

```bash
.venv/bin/agy-token-viz --viz-heuristic-only -- -p "Explain src/"
.venv/bin/jetski-token-viz --viz-heuristic-only -- -p "Explain src/"
.venv/bin/token-viz --viz-target jetski --viz-heuristic-only -- -p "Explain src/"
```

---

## Activity Categories

By default, each `PLANNER_RESPONSE` step is classified into one of 6 categories:

| Category | Bar Label | What It Includes |
| :--- | :--- | :--- |
| **`Planning`** | `Plan` | Proposing an implementation plan/roadmap before making changes, writing `implementation_plan.md`, defining subagents (`define_subagent`), or pure internal reasoning steps with no tools or user output. |
| **`Exploration`** | `Explore` | File/codebase/web/board reads (`read_file`, `read_board`, `grep_search`, `find_by_name`, `list_dir`, `search_web`, `swarm` `status`/`models`) **and** read-only shell inspection via `run_command` (`ls`, `find`, `rg`, `cs`, `cat`, `head`, `git`/`g4`/`p4`/`hg` `status`/`diff`/`log`, `blaze`/`bazel query`) without file output redirection. |
| **`Coding`** | `Code` | Creating, editing, or patching files (`write_file`, `edit_file`, `multi_edit`, `apply_patch`). |
| **`Commands`** | `Cmd` | Non-test shell execution (`blaze build`/`run`, `npm run build`, `pip install`, scripts, `git`/`g4`/`hg` mutations, redirects) and background task / multi-agent orchestration (`manage_task`, `manage_subagents`, `schedule`, `run_workflow`, `swarm`, `send_message`). |
| **`Testing`** | `Test` | Running test suites, linters, or typecheckers via `run_command` (`pytest`, `blaze test`, `bazel coverage`, `tricorder`, `python -m unittest`, `npm test`, `cargo test`/`clippy`, `go test`, `mypy`, `ruff`, `eslint`, `tsc`). |
| **`Communication`** | `Comm` | Replying directly to the user with explanations, architecture walkthroughs, summaries of completed work, or clarifying questions (`ask_question`). |
| **`Harness Overhead`** | `Harness` | Step 1 system preamble (~7k–10.5k tokens of `agy`/`jetski` instructions, tool declarations, and skill metadata) isolated from the user's initial prompt. |

### Configuration

Settings resolve in order: **CLI flags** > **environment variables** (`JETSKI_VIZ_*` / `AGY_VIZ_*` / `TOKEN_VIZ_*`) > **JSON file** (`~/.config/agy-token-viz/config.json` or `--viz-config`) > **defaults**.

| CLI Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--visualize-tokens` | — | *off via `bin/agy` & `bin/jetski`* | Enable live visualization and exit summary |
| `--viz-target` | `TOKEN_VIZ_TARGET` | *(inferred: `agy` or `jetski`)* | Target CLI harness (`agy` or `jetski`) |
| `--viz-model` | `AGY_VIZ_MODEL` / `JETSKI_VIZ_MODEL` | `qwen2.5-3b` | Preset (`qwen2.5-3b`, `qwen2.5-7b`, `llama-3.2-3b`, `llama-3.1-8b`, `qwen2.5-1.5b`, `llama-3.2-1b`, `qwen2.5-0.5b`, `smollm2-360m`), `<org>/<repo>:<file.gguf>`, or `.gguf` path |
| `--viz-model-file` | `AGY_VIZ_MODEL_FILE` / `JETSKI_VIZ_MODEL_FILE` | *(preset default)* | Override GGUF filename within a Hugging Face repo |
| `--viz-model-dir` | `AGY_VIZ_MODEL_DIR` / `JETSKI_VIZ_MODEL_DIR` | `~/.cache/agy-token-viz/models` | Shared directory where GGUF weights are stored and read from |
| `--viz-categories` | `AGY_VIZ_CATEGORIES` / `JETSKI_VIZ_CATEGORIES` | `Planning,Exploration,Coding,Commands,Testing,Communication` | Comma-separated classification labels |
| `--viz-heuristic-only` | `AGY_VIZ_HEURISTIC_ONLY` / `JETSKI_VIZ_HEURISTIC_ONLY` | `false` | Skip GGUF loading; classify purely via tool/command/keyword rules |
| `--viz-include-history` | — | `false` | Count existing transcript steps on `--continue` / `--conversation` / `--conversation_path` |
| `--viz-preload-only` | — | `false` | Download/verify weights, test router load, and exit |
| `--viz-config` | `AGY_VIZ_CONFIG` / `JETSKI_VIZ_CONFIG` | `~/.config/agy-token-viz/config.json` | Optional JSON config (`model`, `model_file`, `model_dir`, `categories`, `agy_data_dir`, `jetski_data_dir`, `n_gpu_layers`, `n_threads`) |
| `--viz-data-dir` | `TOKEN_VIZ_DATA_DIR` / `AGY_DATA_DIR` / `JETSKI_DATA_DIR` | `~/.gemini/antigravity-cli` (`agy`) or `~/.gemini/jetski` (`jetski`) | Harness app data root (also respects forwarded `--gemini_dir` / `--app_data_dir`) |
| `--viz-bin` / `--viz-agy-bin` / `--viz-jetski-bin` | `TOKEN_VIZ_REAL_BIN` / `AGY_REAL_BIN` / `JETSKI_REAL_BIN` | *(auto-discovered)* | Path to the underlying `agy` or `jetski` executable |
| — | `AGY_VIZ_GPU_LAYERS` / `JETSKI_VIZ_GPU_LAYERS` | `-1` | Layers offloaded to Metal/GPU (`-1` = all) |
| — | `AGY_VIZ_THREADS` / `JETSKI_VIZ_THREADS` | `4` | CPU threads passed to `llama.cpp` |
| — | `TOKEN_VIZ_PYTHON` / `AGY_VIZ_PYTHON` / `JETSKI_VIZ_PYTHON` | `<repo>/.venv/bin/python` | Python interpreter invoked by `bin/agy` and `bin/jetski` |
| — | `HF_TOKEN` | *(unset)* | Hugging Face token (also checks `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`) |
| — | `NO_COLOR` | *(unset)* | Disable ANSI color sequences when non-empty (or when `TERM=dumb`) |

---

## Key Design Decisions

- **Local SentencePiece tokenization & `Planning` attribution (`tokenizer.py`, `transcript_watcher.py`, `state.py`)**:
  - Both `agy` and `jetski` fold Google model `thoughtsTokenCount` into `output_tokens` and log summarized/redacted `thinking` text, while storing `content` and `tool_calls` verbatim in `transcript_full.jsonl`. `parse_planner_step` encodes the verbatim `content` + serialized `tool_calls` using a bundled 256,000-vocab Gemini/Gemma SentencePiece model (`sentencepiece`) and attributes the residual `thinking_tokens = max(0, output_tokens - visible_output_tokens)` to `Planning` (and surfaces `Think:` in the status bar and summary), while routing `visible_output_tokens` + `net_input_tokens` to the step's action category.
- **Step 1 `Harness Overhead` isolation (`transcript_watcher.py`, `state.py`)**:
  - On Step 1 of a fresh session (`cache_read_tokens == 0`), the visualizer tokenizes the initial user query with the local SentencePiece model and isolates the ~7k–10.5k token `agy` / `jetski` system preamble (`step_1_input_tokens - user_prompt_tokens`) into `Harness Overhead`.
- **Net context deltas + pruned turn-boundary payload tracking (`state.py`)**:
  - Raw API `input_tokens` per call reports *uncached* prompt tokens (`prompt_tokens - cache_read_tokens`), which re-counts ~2k–25k uncached prefix-cache block tails on small turns and re-bills `output_tokens` from step `N` as uncached `input_tokens` on step `N + 1`. `SessionTokenState` instead computes monotonic net context growth per step, and when `agy` / `jetski` / Gemini strips prior thinking blocks across turn or error boundaries (`P_k < prev_end_context`), it directly tokenizes the newly appended event payload (`USER_INPUT`, `GENERIC` tool outputs, `SYSTEM_MESSAGE`, `ERROR_MESSAGE`) via SentencePiece so new input is never masked.
  - Because tool outputs (e.g., file contents from `read_file`) enter the prompt on the *following* turn, net context growth when `not preceded_by_user_input` is attributed back to the preceding tool-invoking step rather than charging the subsequent reply turn.
- **Hybrid tool-signature guardrails + GBNF-constrained GGUF routing (`router.py`)**:
  - Steps whose tools map to a single unambiguous domain (e.g., pure `read_file` → `Exploration`, `ls`/`find`/`g4 status`/`blaze query` → `Exploration`, `pytest`/`blaze test` → `Testing`, `write_file` → `Coding`) resolve deterministically so `read_file(tests/test_foo.py)` or user explanations mentioning tests never leak into `Testing`/`Commands`.
  - Multi-domain tool turns, plan-vs-reply turns, custom `--viz-categories`, and context-dependent MCP/subagent/swarm tools invoke local `llama.cpp` (`Qwen2.5-3B-Instruct` Q4_K_M by default; `qwen2.5-7b` and `llama-3.1-8b` also built in) constrained by a per-step GBNF `LlamaGrammar` (`temperature=0.0`) so output is guaranteed to be a valid candidate label.
- **Zero-overhead pass-through shims (`bin/agy`, `bin/jetski`, `bin/_token_viz_shim.sh`)**: Without `--visualize-tokens` or `--viz-*` flags (or when running non-session subcommands like `models`, `web`, `chatski`, `--stamp`, or `--help`), the Bash wrappers resolve the real binary via `realpath` and `exec` it directly without starting Python.
- **Per-process CLI-log binding & out-of-band transcript tailing (`transcript_watcher.py`)**: Binds to the spawned child PID via `<DATA_DIR>/log/cli-*.log` (`agy`) or `<DATA_DIR>/cli/log/cli-*.log` (`jetski`), excludes pre-existing/concurrent sessions on fresh runs, tracks spawned subagent and swarm worker conversations, resets state on interactive `/new` conversations, and polls `.system_generated/logs/transcript.jsonl` at 150ms byte-offset cursors (hydrating truncated lines on-demand from `transcript_full.jsonl`).
- **PTY scroll-region, alt-screen & synchronized-output framing (`pty_runner.py`)**: Interactive runs spawn `agy` or `jetski` in a `(rows - 2, cols)` PTY confined via ANSI `DECSTBM` (`\x1b[1;{rows-2}r`) across both normal and alternate (`\x1b[?1049h`) screen buffers, resetting SGR before footer line erases to prevent 24-bit RGB Background Color Erase bleed, injecting footer repaints atomically inside DEC Mode 2026 (`\x1b[?2026h`…`\x1b[?2026l`) synchronized frames, and clamping/filtering SGR mouse events over the bottom 2 rows. Piped runs bypass the PTY to keep `stdout` clean.

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a detailed walkthrough of why API `Billed` tokens differ from visualized tokens when running Google models via `agy` or `jetski`.

- **Residual thinking-token wire framing**: Unless `transcript.jsonl` provides an explicit `thinking_tokens` field, `thinking_tokens` is computed as `output_tokens - count_tokens_locally(content + serialized_tool_calls)` using the Gemini/Gemma SentencePiece vocabulary; minor protobuf function-call control framing (~5–30 tokens per tool call) can shift the residual slightly.
- **Unix / POSIX terminals only**: Depends on `bash`, `pty`, `termios`, `tty`, `fcntl` (`TIOCGWINSZ`/`TIOCSWINSZ`), and `SIGWINCH`. Native Windows consoles are unsupported (use WSL).
- **Coupled to harness transcript & CLI log schemas**: Relies on `agy` / `jetski` writing completed `PLANNER_RESPONSE` (`status: "DONE"`) JSONL entries under `<DATA_DIR>/brain/<id>/.system_generated/logs/transcript.jsonl` and process/conversation logs under `<DATA_DIR>/log/cli-*.log` or `<DATA_DIR>/cli/log/cli-*.log`. Note that `jetski --isolated_env` without `--cli_log_file` stores its temporary data directory outside `~/.gemini/jetski` unless passed via `--viz-data-dir` or `--cli_log_file`.
- **Single action label per step**: Steps update on completion (`status == "DONE"`), not token-by-token mid-stream. While `thinking_tokens` are split into `Planning` and Step 1 preamble into `Harness Overhead`, a step's remaining visible/input tokens are assigned to a single action category.
- **Custom category prompts**: Custom `--viz-categories` labels that do not match built-in domain keywords bypass tool guardrails and use generic system-prompt descriptions (`"Activity related to <Category>."`) without tailored few-shot examples.
- **Ephemeral state & terminal height**: Ledger state is held in memory for the current run only. Terminals shorter than 6 rows disable the live footer, and category styles cycle after 7 categories.
