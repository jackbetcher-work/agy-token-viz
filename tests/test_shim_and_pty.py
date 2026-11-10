"""Integration tests for `bin/agy` Bash shim and PTY wrapper (`pty_runner.py`)."""

from __future__ import annotations

import os
from pathlib import Path
import pty
import select
import stat
import subprocess
import time

from agy_token_viz.pty_runner import split_complete_terminal_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_AGY = PROJECT_ROOT / "bin" / "agy"


def test_split_complete_terminal_bytes_preserves_incomplete_sequences() -> None:
    # Incomplete CSI sequence at end
    prefix, tail = split_complete_terminal_bytes(b"hello\x1b[32;1")
    assert prefix == b"hello"
    assert tail == b"\x1b[32;1"

    # Complete CSI sequence
    prefix2, tail2 = split_complete_terminal_bytes(b"hello\x1b[32;1mworld")
    assert prefix2 == b"hello\x1b[32;1mworld"
    assert tail2 == b""

    # Incomplete 3-byte UTF-8 block character '█' (\xe2\x96\x88)
    block = "█".encode("utf-8")
    p_utf, t_utf = split_complete_terminal_bytes(b"bar:" + block[:2])
    assert p_utf == b"bar:"
    assert t_utf == block[:2]


def _make_fake_agy_script(tmp_path: Path) -> Path:
    fake_agy = tmp_path / "fake_agy"
    fake_agy.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys, time

print("FAKE_AGY_ARGS:" + json.dumps(sys.argv[1:]), flush=True)
data_dir = os.environ.get("AGY_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "sim-conv-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 3000,
        "cache_read_tokens": 7000,
        "output_tokens": 500,
        "thinking": "Reading project README",
        "tool_calls": [{"name": "read_file", "args": {"Path": "README.md"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")
    time.sleep(0.25)
""",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IXUSR)
    return fake_agy


def test_bash_shim_passthrough_without_visualize_flag(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy_script(tmp_path)
    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_agy),
        "AGY_DATA_DIR": str(tmp_path / "agy_data"),
    }
    proc = subprocess.run(
        [str(BIN_AGY), "-p", "hello world", "--model", "pro"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_AGY_ARGS:["-p", "hello world", "--model", "pro"]' in proc.stdout
    assert "[agy-token-viz]" not in proc.stderr


def test_bash_shim_with_visualize_tokens_strips_flag_and_summarizes(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy_script(tmp_path)
    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_agy),
        "AGY_DATA_DIR": str(tmp_path / "agy_data"),
    }
    proc = subprocess.run(
        [
            str(BIN_AGY),
            "--visualize-tokens",
            "--viz-heuristic-only",
            "-p",
            "inspect repo",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # Real/fake agy must NOT receive --visualize-tokens or --viz-heuristic-only
    assert 'FAKE_AGY_ARGS:["-p", "inspect repo"]' in proc.stdout
    assert "Session Token Summary" in proc.stderr
    assert "Exploration" in proc.stderr
    assert "3.5k" in proc.stderr


def test_pty_interactive_mode_renders_live_2_line_bar(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy_script(tmp_path)
    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_agy),
        "AGY_DATA_DIR": str(tmp_path / "agy_data"),
        "TERM": "xterm-256color",
    }

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        [str(BIN_AGY), "--visualize-tokens", "--viz-heuristic-only", "-p", "hi"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    captured = bytearray()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(master_fd, 8192)
                if not chunk:
                    break
                captured.extend(chunk)
            except OSError:
                break
        elif proc.poll() is not None:
            break

    os.close(master_fd)
    assert proc.wait(timeout=5) == 0
    rendered_text = captured.decode("utf-8", errors="replace")
    assert "FAKE_AGY_ARGS:" in rendered_text
    assert "Tokens" in rendered_text
    assert "Session Token Summary" in rendered_text
