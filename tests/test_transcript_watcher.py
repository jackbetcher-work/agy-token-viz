"""Unit tests for real-time Antigravity transcript watching (`transcript_watcher.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from agy_token_viz.state import Category, SessionTokenState, StepRecord
from agy_token_viz.tokenizer import count_tokens_locally, serialize_tool_calls_verbatim
from agy_token_viz.transcript_watcher import (
    TranscriptWatcher,
    extract_log_file_from_args,
    extract_target_conversation_from_args,
    is_resuming_conversation_from_args,
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
    assert by_name[Category.PLANNING].output_tokens == exact_thought_tokens if (exact_thought_tokens := actual_thought_tokens) else True


def test_new_session_excludes_preexisting_and_concurrent_conversations(tmp_path: Path) -> None:
    """
    When invoking a new `agy` command (`exclude_preexisting=True`, `lock_to_first_conversation=True`),
    existing conversations (including sessions actively writing steps in another terminal) must not
    leak their token state into the new session.
    """
    other_logs = _make_logs_dir(tmp_path, "conv-other-terminal")
    other_transcript = other_logs / "transcript.jsonl"
    other_transcript.write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 8000,
                "cache_read_tokens": 0,
                "output_tokens": 400,
                "content": "Initial step in other terminal",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen: list[StepRecord] = []
    watcher = TranscriptWatcher(
        agy_data_dir=tmp_path,
        on_step=seen.append,
        exclude_preexisting=True,
        lock_to_first_conversation=True,
    )
    watcher.capture_baseline()

    # Concurrent terminal appends Step 2 to its pre-existing conversation after our baseline
    with other_transcript.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "step_index": 2,
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "input_tokens": 6000,
                    "cache_read_tokens": 48000,
                    "output_tokens": 300,
                    "content": "Concurrent step in other terminal",
                }
            )
            + "\n"
        )

    assert watcher.poll_once() == 0
    assert seen == []

    # Now our new session creates its own conversation directory and logs Step 1
    own_logs = _make_logs_dir(tmp_path, "conv-own-new")
    own_transcript = own_logs / "transcript.jsonl"
    own_transcript.write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 8100,
                "cache_read_tokens": 0,
                "output_tokens": 150,
                "content": "Hello from new session",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert watcher.poll_once() == 1
    assert len(seen) == 1
    assert seen[0].conversation_id == "conv-own-new"

    # A third conversation created later in another terminal is also ignored once locked
    third_logs = _make_logs_dir(tmp_path, "conv-third-terminal")
    (third_logs / "transcript.jsonl").write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 9000,
                "output_tokens": 200,
                "content": "Third terminal step",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert watcher.poll_once() == 0
    assert len(seen) == 1


def test_cli_log_pid_binding_and_new_conversation_reset(tmp_path: Path) -> None:
    """
    Verify that binding `TranscriptWatcher` to the spawned `agy` PID via `log/cli-*.log`
    ignores unrelated conversations created after baseline and resets state on `/new`.
    """
    log_dir = tmp_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    cli_log = log_dir / "cli-20261110_163007.log"
    cli_log.write_text(
        "I1110 16:30:07.918086 55 server.go:2341] Starting language server process with pid 42424\n",
        encoding="utf-8",
    )

    state = SessionTokenState()
    watcher = TranscriptWatcher(
        agy_data_dir=tmp_path,
        on_step=lambda step: state.record_classified_step(step, Category.COMMUNICATION),
        exclude_preexisting=True,
        lock_to_first_conversation=True,
        on_reset=state.reset,
    )
    watcher.capture_baseline()
    watcher.set_child_pid(42424)

    # Another terminal creates a brand-new conversation while our `agy` (pid 42424) is still idle at prompt
    unrelated_id = "aaaaaaaa-1111-2222-3333-444444444444"
    unrelated_logs = _make_logs_dir(tmp_path, unrelated_id)
    (unrelated_logs / "transcript.jsonl").write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 9000,
                "output_tokens": 500,
                "content": "Unrelated new conversation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert watcher.poll_once() == 0
    assert state.snapshot().total_active_tokens == 0

    # Now pid 42424 creates its first conversation in cli.log
    conv1_id = "11111111-2222-3333-4444-555555555555"
    with cli_log.open("a", encoding="utf-8") as fh:
        fh.write(f"I1110 16:30:20.000000 88 server.go:1233] Created conversation {conv1_id}\n")

    conv1_logs = _make_logs_dir(tmp_path, conv1_id)
    (conv1_logs / "transcript.jsonl").write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 4000,
                "output_tokens": 300,
                "content": "First conversation reply",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert watcher.poll_once() == 1
    assert state.snapshot().total_active_tokens == 4300

    # User starts a `/new` conversation inside the same `agy` process -> state resets cleanly
    conv2_id = "66666666-7777-8888-9999-000000000000"
    with cli_log.open("a", encoding="utf-8") as fh:
        fh.write(f"I1110 16:31:00.000000 88 server.go:1233] Created conversation {conv2_id}\n")

    conv2_logs = _make_logs_dir(tmp_path, conv2_id)
    (conv2_logs / "transcript.jsonl").write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 2000,
                "output_tokens": 100,
                "content": "Second conversation fresh start",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert watcher.poll_once() == 1
    assert state.snapshot().total_steps == 1
    assert state.snapshot().total_active_tokens == 2100


def test_resumed_conversation_seeds_baseline_context_without_old_tokens(tmp_path: Path) -> None:
    """
    When resuming via `--continue` (`include_history=False`), prior context size is seeded
    so the session starts at 0 tokens and Step N+1 only counts net new prompt growth.
    """
    assert is_resuming_conversation_from_args(["--continue"]) is True
    assert is_resuming_conversation_from_args(["-c"]) is True
    assert is_resuming_conversation_from_args(["--conversation", "abc"]) is True
    assert is_resuming_conversation_from_args(["-p", "hi"]) is False
    assert extract_log_file_from_args(["--log-file", "/tmp/a.log"]) == Path("/tmp/a.log")

    conv_id = "conv-resumed"
    logs_dir = _make_logs_dir(tmp_path, conv_id)
    transcript = logs_dir / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 8000,
                "cache_read_tokens": 40000,
                "output_tokens": 500,
                "content": "Prior session step",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = SessionTokenState()
    watcher = TranscriptWatcher(
        agy_data_dir=tmp_path,
        on_step=lambda step: state.record_classified_step(step, Category.COMMUNICATION),
        exclude_preexisting=False,
        on_baseline_context=lambda cid, p, v, e: state.seed_conversation_baseline(
            cid,
            prev_prompt_tokens=p,
            prev_visible_end_context=v,
            prev_end_context=e,
        ),
    )
    watcher.capture_baseline()

    # Before new step: state is completely clean (0 tokens)
    assert state.snapshot().total_active_tokens == 0

    # Step 2 in resumed session: prior end_context was 48,500; new prompt is 45,000 cache + 3,800 in = 48,800
    # Only the 300 net new input tokens + 200 output tokens = 500 active tokens should be recorded!
    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "step_index": 2,
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "input_tokens": 3800,
                    "cache_read_tokens": 45000,
                    "output_tokens": 200,
                    "content": "Resumed reply",
                }
            )
            + "\n"
        )

    assert watcher.poll_once() == 1
    snap = state.snapshot()
    assert snap.total_input_tokens == 300
    assert snap.total_output_tokens == 200
    assert snap.total_active_tokens == 500


def test_fresh_session_skips_preexisting_inspection_and_globbing_once_bound(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Verify:
    1. On a fresh run (`exclude_preexisting=True`), `capture_baseline()` does not parse
       existing transcripts in `brain/`.
    2. Once bound to an allowed conversation, `poll_once()` checks only that conversation's
       transcript path without globbing `brain/*`.
    """
    for idx in range(5):
        old_dir = _make_logs_dir(tmp_path, f"old-conv-{idx}")
        (old_dir / "transcript.jsonl").write_text(
            json.dumps(
                {
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "input_tokens": 5000,
                    "output_tokens": 200,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    seen: list[StepRecord] = []
    watcher = TranscriptWatcher(
        agy_data_dir=tmp_path,
        on_step=seen.append,
        exclude_preexisting=True,
        lock_to_first_conversation=True,
    )

    inspect_calls: list[str] = []
    orig_inspect = watcher._inspect_existing_transcript

    def _tracked_inspect(path: Path, conv_id: str):
        inspect_calls.append(conv_id)
        return orig_inspect(path, conv_id)

    monkeypatch.setattr(watcher, "_inspect_existing_transcript", _tracked_inspect)
    watcher.capture_baseline()
    assert inspect_calls == []

    # Create active conversation and lock onto it on first poll
    active_dir = _make_logs_dir(tmp_path, "active-conv")
    active_transcript = active_dir / "transcript.jsonl"
    active_transcript.write_text(
        json.dumps(
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "input_tokens": 4000,
                "output_tokens": 150,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert watcher.poll_once() == 1

    # Once locked onto `active-conv`, subsequent polls must not call `_iter_transcript_paths`
    iter_calls: list[int] = []
    monkeypatch.setattr(
        watcher,
        "_iter_transcript_paths",
        lambda: iter_calls.append(1) or [],  # type: ignore[func-returns-value]
    )
    with active_transcript.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "step_index": 2,
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "input_tokens": 4500,
                    "output_tokens": 200,
                }
            )
            + "\n"
        )
    assert watcher.poll_once() == 1
    assert iter_calls == []
    assert len(seen) == 2

