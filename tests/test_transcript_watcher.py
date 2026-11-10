"""Unit tests for real-time Antigravity transcript watching (`transcript_watcher.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from agy_token_viz.state import Category, SessionTokenState, StepRecord
from agy_token_viz.tokenizer import count_tokens_locally, serialize_tool_calls_verbatim
from agy_token_viz.transcript_watcher import (
    TranscriptWatcher,
    extract_target_conversation_from_args,
    parse_planner_step,
)


def _make_logs_dir(agy_data_dir: Path, conv_id: str) -> Path:
    logs_dir = agy_data_dir / "brain" / conv_id / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def test_baseline_ignores_prior_history_and_tails_new_steps(tmp_path: Path) -> None:
    logs_dir = _make_logs_dir(tmp_path, "conv-existing")
    transcript = logs_dir / "transcript.jsonl"
    old_step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 5000,
        "cache_read_tokens": 0,
        "output_tokens": 300,
        "thinking": "Old step from yesterday",
    }
    transcript.write_text(json.dumps(old_step) + "\n", encoding="utf-8")

    seen: list[StepRecord] = []
    watcher = TranscriptWatcher(agy_data_dir=tmp_path, on_step=seen.append)
    watcher.capture_baseline()

    # Initial poll should NOT emit old_step
    assert watcher.poll_once() == 0
    assert seen == []

    # Append partial JSON line without newline -> must wait until newline arrives
    new_step = {
        "step_index": 2,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 2200,
        "cache_read_tokens": 5000,
        "output_tokens": 410,
        "thinking": "Running pytest",
        "tool_calls": [{"name": "run_command", "args": {"CommandLine": "pytest"}}],
    }
    serialized = json.dumps(new_step)
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(serialized[:25])

    assert watcher.poll_once() == 0
    assert seen == []

    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(serialized[25:] + "\n")

    assert watcher.poll_once() == 1
    assert len(seen) == 1
    assert seen[0].step_index == 2
    assert seen[0].input_tokens == 2200
    assert seen[0].output_tokens == 410
    assert seen[0].tool_calls[0].name == "run_command"
    assert "pytest" in seen[0].tool_calls[0].summary


def test_truncated_fields_hydrates_from_transcript_full(tmp_path: Path) -> None:
    logs_dir = _make_logs_dir(tmp_path, "conv-trunc")
    transcript = logs_dir / "transcript.jsonl"
    transcript_full = logs_dir / "transcript_full.jsonl"

    seen: list[StepRecord] = []
    watcher = TranscriptWatcher(agy_data_dir=tmp_path, on_step=seen.append)
    watcher.capture_baseline()

    compact_line = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 1000,
        "cache_read_tokens": 0,
        "output_tokens": 250,
        "content": "<truncated>",
        "truncated_fields": ["content"],
    }
    full_line = {
        **compact_line,
        "content": "Full untruncated explanation of the code changes.",
    }
    full_line.pop("truncated_fields", None)

    transcript_full.write_text(json.dumps(full_line) + "\n", encoding="utf-8")
    transcript.write_text(json.dumps(compact_line) + "\n", encoding="utf-8")

    assert watcher.poll_once() == 1
    assert seen[0].content == "Full untruncated explanation of the code changes."


def test_extract_target_conversation_from_args() -> None:
    assert extract_target_conversation_from_args(["-p", "hi"]) is None
    assert extract_target_conversation_from_args(["--conversation", "abc-123"]) == "abc-123"
    assert extract_target_conversation_from_args(["--conversation=xyz-789"]) == "xyz-789"


def test_watcher_tracks_preceded_by_user_input_flag(tmp_path: Path) -> None:
    logs_dir = _make_logs_dir(tmp_path, "conv-user-flag")
    transcript = logs_dir / "transcript.jsonl"
    seen: list[StepRecord] = []
    watcher = TranscriptWatcher(agy_data_dir=tmp_path, on_step=seen.append)
    watcher.capture_baseline()

    events = [
        {"step_index": 0, "type": "USER_INPUT", "status": "DONE", "content": "hi"},
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 1000,
            "output_tokens": 50,
            "tool_calls": [{"name": "read_file", "args": {"Path": "a.py"}}],
        },
        {"step_index": 2, "type": "GENERIC", "status": "DONE", "content": "file contents"},
        {
            "step_index": 3,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 1200,
            "output_tokens": 80,
            "content": "Done reading",
        },
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    assert watcher.poll_once() == 2
    assert seen[0].preceded_by_user_input is True
    assert seen[1].preceded_by_user_input is False


def test_parse_planner_step_estimates_thinking_tokens_from_summarized_or_empty_thinking() -> None:
    # 1. Explicit thinking_tokens field in payload takes precedence
    explicit = parse_planner_step(
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 1000,
            "output_tokens": 900,
            "thinking_tokens": 700,
            "content": "Short reply.",
        },
        "conv-explicit",
    )
    assert explicit is not None
    assert explicit.thinking_tokens == 700
    assert explicit.visible_output_tokens == 200

    # 2. Summarized/placeholder `thinking` on a Google model step: residual from verbatim visible output
    summarized = parse_planner_step(
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 5000,
            "output_tokens": 2500,
            "thinking": "I'm thinking through how to approach this.\n\n\n",
            "content": "",
            "tool_calls": [{"name": "read_file", "args": {"Path": "src/main.py"}}],
        },
        "conv-summary",
    )
    assert summarized is not None
    assert summarized.visible_output_tokens < 50
    assert summarized.thinking_tokens > 2450
    assert summarized.thinking_tokens + summarized.visible_output_tokens == 2500

    # 3. Empty `thinking` ("") on a tool step where `output_tokens` far exceeds visible tool JSON
    redacted = parse_planner_step(
        {
            "step_index": 3,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 5000,
            "output_tokens": 1200,
            "thinking": "",
            "content": "",
            "tool_calls": [{"name": "run_command", "args": {"CommandLine": "pytest -q"}}],
        },
        "conv-redacted",
    )
    assert redacted is not None
    assert redacted.thinking_tokens > 1100

    # 4. Empty `thinking` ("") where `output_tokens` matches visible `content` -> 0 thinking tokens
    no_think_text = "The quick brown fox jumps over the lazy dog. " * 10
    exact_vis = count_tokens_locally(no_think_text)
    no_think = parse_planner_step(
        {
            "step_index": 4,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 1000,
            "output_tokens": exact_vis + 2,
            "thinking": "",
            "content": no_think_text,
        },
        "conv-no-think",
    )
    assert no_think is not None
    assert no_think.thinking_tokens == 0
    assert no_think.visible_output_tokens == exact_vis + 2


def test_large_file_write_step_uses_sentencepiece_without_thinking_token_leak() -> None:
    """
    Acceptance Criterion 1: Replay a large (~28 KB) dense code/CSS `write_file` step.
    SentencePiece accurately tokenizes the dense code (~12,450 tokens) so `thinking_tokens`
    does not artificially spike (unlike `chars / 4.0` which underestimated by >5,000 tokens).
    """
    dense_css_block = (
        ":root { --gcp-blue: #1a73e8; --surface-0: #0b0f19; --border: rgba(255,255,255,0.08); }\n"
        ".card-grid > .item[data-active='true'] { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }\n"
    ) * 145  # ~28.4 KB of dense CSS/punctuation
    raw_tools = [
        {
            "name": "write_file",
            "args": {"Path": "/repo/styles/main.css", "Content": dense_css_block},
        }
    ]
    serialized = serialize_tool_calls_verbatim(raw_tools)
    exact_code_tokens = count_tokens_locally(serialized)
    actual_thought_tokens = 120
    api_output_tokens = exact_code_tokens + actual_thought_tokens

    step = parse_planner_step(
        {
            "step_index": 18,
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "input_tokens": 5000,
            "cache_read_tokens": 50000,
            "output_tokens": api_output_tokens,
            "thinking": "Designing a minimalist CSS stylesheet.",
            "content": "",
            "tool_calls": raw_tools,
        },
        "conv-large-write",
    )
    assert step is not None
    assert step.thinking_tokens == actual_thought_tokens
    assert step.visible_output_tokens == exact_code_tokens

    state = SessionTokenState()
    state.record_classified_step(step, Category.CODING)
    by_name = {c.name: c for c in state.snapshot().categories}
    assert by_name[Category.CODING].output_tokens == exact_code_tokens
    assert by_name[Category.PLANNING].output_tokens == actual_thought_tokens
