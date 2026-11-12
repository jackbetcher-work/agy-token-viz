# Token Visualization & Billing Limitations (`agy` Harness + Google Models)

This document explains why **API-billed token totals** (`Billed` input, `Cache`, and `Out`) differ from the **net active tokens** (`Tokens` / `Active`) and per-category breakdowns displayed by `agy-token-viz` (`agy --visualize-tokens`) when running Google Gemini models through the Antigravity CLI (`agy`) harness.

---

## 1. Overview: Net Active Context vs. Cumulative API Billing

An agent turn in `agy` often spans 5–50+ sequential LLM API calls (`PLANNER_RESPONSE` steps). On every step `k`, `agy` sends the entire conversation history up to that point as the prompt and logs three counters in `<AGY_DATA_DIR>/brain/<id>/.system_generated/logs/transcript.jsonl`:

| Transcript Field | What the API Reports on Step `k` |
| :--- | :--- |
| `cache_read_tokens` (`C_k`) | Prefix tokens served from Google's context cache on step `k`. |
| `input_tokens` (`I_k`) | **Uncached** prompt tokens billed at full input rate on step `k` (`P_k - C_k`, where `P_k = C_k + I_k` is total prompt size). |
| `output_tokens` (`O_k`) | Total generated tokens billed at output rate on step `k` (**thinking tokens + visible response/tool tokens**). |

If a visualizer simply summed `I_k + O_k` across every step, the same prompt tokens would be counted repeatedly whenever they sit in the uncached tail of the prompt. Instead, `agy-token-viz` separates **cumulative API billing** from **net active tokens**:

- **Visualized `In` (`total_input_tokens`)**: Monotonic net prompt context growth across turns (`max(0, P_k - prev_end_context)`), falling back to local SentencePiece tokenization of newly appended payloads (`USER_INPUT`, `GENERIC` tool outputs, `SYSTEM_MESSAGE`, `ERROR_MESSAGE`) whenever prior thinking blocks are pruned across turn boundaries (`P_k < prev_end_context`).
- **Step 1 `Harness Overhead` (`total_harness_overhead_tokens`)**: On Step 1 (`cache_read_tokens == 0`), `agy-token-viz` tokenizes the initial user query with the local Gemini SentencePiece tokenizer (`tokenizer.py`) and isolates the ~7,000–9,500 token `agy` system preamble (`step_1_input_tokens - user_prompt_tokens`) into `Harness Overhead`.
- **Summary `Billed` (`total_billed_input_tokens`)**: Raw `∑ I_k` billed by the API across all steps.
- **Visualized `Cache` (`total_cache_read_tokens`)**: Cumulative `∑ C_k` read from prefix cache across all steps.
- **Visualized `Out` (`total_output_tokens`) & `Think` (`total_thinking_tokens`)**: Cumulative `∑ O_k`, split into `visible_output_tokens` (computed by local SentencePiece tokenization of `content` + verbatim `tool_calls`, routed to the step's action category) and residual `thinking_tokens = max(0, output_tokens - visible_output_tokens)` (routed to `Planning`).

Because of how the `agy` harness and Google models manage context caching and chain-of-thought reasoning, several structural discrepancies remain between what the API bills and what is displayed in the visualization.

---

## 2. Uncached Prefix-Cache Block Tails (`Billed` Input > Visualized `In`)

Google's context cache operates on coarse block boundaries and does not immediately cache the trailing tokens of the most recent step(s). As a result, on every step `k` in a multi-step tool loop, `input_tokens` (`I_k`) includes:

1. **Truly new context** added since step `k - 1` (e.g., a `read_file` result or user message), **plus**
2. **Uncached tail tokens from earlier steps** (~2,000–25,000 tokens) that have not yet crossed a prefix-cache boundary, **plus**
3. **`output_tokens` from step `k - 1`** (`O_k-1`), which are now part of the conversation history on step `k` and are billed again as uncached input before entering the cache on step `k + 1` or `k + 2`.

### Concrete Example from an `agy` Transcript

| Step | Action | `cache_read_tokens` (`C_k`) | API `input_tokens` (`I_k`) | Total Prompt (`P_k`) | API `output_tokens` (`O_k`) | Visualized Net `In` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | Harness preamble (`7,928`) + User prompt (`118`) | `0` | `8,046` | `8,046` | `277` | `7,928` (`Harness`) + `118` (`Explore`) |
| **2** | `ls` output (`+626` tok) + 8× `read_file` | `5,992` | `2,957` | `8,949` | `440` | `626` (`Explore`) |
| **3** | 8 file reads (`+12,252` tok) + 5× `read_file` | `8,950` | `12,691` | `21,641` | `310` | `12,252` (`Explore`) |

- On **Step 2**, only `5,992` of Step 1's `8,046` prompt tokens hit the prefix cache. The remaining `2,054` tokens from Step 1—along with Step 1's `277` output tokens—were re-billed inside Step 2's `input_tokens = 2,957`, even though the `ls` command output only added `626` net tokens.
- Across a 40-step session (`9de84d20`), raw API `Billed` input (`∑ I_k`) reaches **`752k`**, whereas true net context input (`In`) is **`112k`**—a **`640k` token difference** caused by uncached block tails and prior outputs being re-billed as uncached input across 40 turns.

---

## 3. Thinking Token Echoing Into Subsequent `Billed` Input & `Cache`

Within an active `agy` tool-calling sequence, the harness passes the model's thought blocks/signatures from Step `k` back into the prompt on Step `k + 1` so the model maintains reasoning continuity across tool calls:

- Suppose Step `k` generates **`24,139` thinking tokens** and **`283` visible tool-call tokens** (`output_tokens = 24,422`), and the tool returns **`180` tokens** of output.
- On Step `k + 1`, `agy` echoes all `24,422` output tokens back in the prompt: API `input_tokens` spikes to **`24,601`** (`24,422` prior output/thinking + `179` tool result).
- On Step `k + 2`, those `24,422` tokens roll into `cache_read_tokens` and are re-read from cache on every remaining step of the session.
- **Result**: A single deep-thinking step inflates `Out` by 24.4k on Step `k`, inflates `Billed` input by 24.4k on Step `k + 1`, and inflates cumulative `Cache` by `24.4k × (remaining steps)`, whereas `agy-token-viz` counts those 24.4k tokens **once** in `Active` (`Out` / `Think`) on Step `k` and attributes only the `180` new tool-result tokens as net `In`.

> **Note:** When `agy`/Gemini strips prior thinking blocks at turn or error boundaries so `P_k < prev_end_context`, [`state.py`](src/agy_token_viz/state.py) falls back to directly tokenizing the newly appended `USER_INPUT` / `GENERIC` / `SYSTEM_MESSAGE` payload via [`tokenizer.py`](src/agy_token_viz/tokenizer.py) so new input is not masked by the negative prompt delta.

---

## 4. Residual `thinking_tokens` Calculation vs. Wire-Format Framing Overhead

Because `agy` does not write Google's `usageMetadata.thoughtsTokenCount` as a separate field in `transcript.jsonl` and logs only summarized or empty (`""`) `thinking` text for Google models, [`transcript_watcher.py`](src/agy_token_viz/transcript_watcher.py) computes `visible_output_tokens` by encoding the verbatim `content` and serialized `tool_calls` with the local 256,000-vocab Gemini/Gemma SentencePiece model ([`tokenizer.py`](src/agy_token_viz/tokenizer.py)) and derives:

```text
thinking_tokens = max(0, output_tokens - visible_output_tokens)
```

- **Minor structural framing difference (~5–30 tokens per tool call)**: Google's internal protobuf function-call framing adds a few special control tokens per tool invocation that are not part of the JSON `tool_calls` array in `transcript_full.jsonl`. On turns where `thinking` is non-empty, those ~5–30 framing tokens are absorbed into `thinking_tokens`; on turns where `thinking` is empty (`""`), a small 8% / 16-token framing margin prevents false-positive thinking attribution, which can leave very brief unlogged thoughts (<16 tokens) inside visible `Out`.
- **Total `Out` is always exact**: Because `visible_output_tokens + thinking_tokens == output_tokens`, any minor framing residual affects only the split between `Planning` (`Think`) and the step's action category—never `total_output_tokens` or `total_active_tokens`.

---

## 5. Post-Completion (`status: "DONE"`) Flushing, Aborted Turns & Compaction

- **No mid-turn streaming counters**: `agy` only writes token counts (`input_tokens`, `cache_read_tokens`, `output_tokens`) when a `PLANNER_RESPONSE` entry reaches `status: "DONE"`. While a Google model is actively generating thousands of thinking tokens mid-turn, the status bar cannot increment until the API call completes.
- **Aborted or crashed turns**: If `agy` is interrupted (`Ctrl-C` / `SIGINT`) while a model call is still in flight—before `agy` flushes the completed `PLANNER_RESPONSE` line to `transcript.jsonl`—tokens consumed by that aborted in-flight request are billed by the API but never recorded in the transcript or visualization.
- **Context compaction (`/compact` / `CHECKPOINT`)**: When a `CHECKPOINT` event shrinks the conversation prompt, `SessionTokenState` synchronizes its context baseline on that step (`net_in = 0`) so compacted history is not re-billed as new input; however, the API still bills the compaction summarization request and the post-compaction uncached prompt inside `Billed`.
