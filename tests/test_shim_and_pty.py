"""Integration tests for `bin/agy` Bash shim and PTY wrapper (`pty_runner.py`)."""

from __future__ import annotations

import os
from pathlib import Path
import pty
import select
import stat
import subprocess
import time

from agy_token_viz.pty_runner import (
    inject_scroll_region_guards,
    split_complete_terminal_bytes,
)


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


def test_no_visualize_tokens_overrides_default_alias_and_help_bypasses(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy_script(tmp_path)
    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_agy),
        "AGY_DATA_DIR": str(tmp_path / "agy_data"),
    }
    # Simulates `alias agy="agy --visualize-tokens"` overridden via `--no-visualize-tokens`
    proc_opt_out = subprocess.run(
        [str(BIN_AGY), "--visualize-tokens", "--no-visualize-tokens", "-p", "quick"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_AGY_ARGS:["-p", "quick"]' in proc_opt_out.stdout
    assert "[agy-token-viz]" not in proc_opt_out.stderr

    # Simulates `agy --help` or `agy models` when aliased to `agy --visualize-tokens`
    proc_models = subprocess.run(
        [str(BIN_AGY), "--visualize-tokens", "models"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_AGY_ARGS:["models"]' in proc_models.stdout
    assert "[agy-token-viz]" not in proc_models.stderr


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


def test_inject_scroll_region_guards_restores_margins_after_1049l() -> None:
    payload = b"\x1b[?1049l\r\x1b[J\nAntigravity CLI 1.5.5\r\n"
    guarded = inject_scroll_region_guards(payload, child_rows=28)
    assert guarded.startswith(b"\x1b[?1049l\x1b7\x1b[1;28r\x1b8\r\x1b[J\nAntigravity CLI 1.5.5")


def test_pty_preserves_native_logo_and_banner_during_background_stdio_redirect(
    tmp_path: Path,
) -> None:
    """
    Regression test: ensure native Antigravity logo and welcome message emitted right
    at startup (across `?1049h` -> `?1049l`) are never swallowed when a background
    model loader redirects fd 1/2 to /dev/null via `os.dup2` during `on_before_spawn`.
    """
    fake_agy = tmp_path / "fake_agy_logo"
    fake_agy.write_text(
        """#!/usr/bin/env python3
import sys, time

# Simulate native agy startup: alternate buffer splash -> normal buffer RGB logo + message
sys.stdout.write(
    "\\x1b[?1049h\\x1b[H\\x1b[2J\\n Welcome to the Antigravity CLI.\\r\\n \\u28f7  Signing in..."
)
sys.stdout.flush()
time.sleep(0.05)
sys.stdout.write(
    "\\x1b[?1049l\\r\\x1b[J\\n"
    "\\x1b[6C\\x1b[48;2;242;146;46m  \\x1b[48;2;240;114;54m  \\x1b[m        "
    "\\x1b[38;2;122;162;247;1mAntigravity CLI 1.5.5\\x1b[m\\n"
    "\\x1b[35D\\x1b[48;2;158;195;69m  \\x1b[m      user@example.com\\r\\n"
)
sys.stdout.flush()
time.sleep(0.1)
""",
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IXUSR)

    runner_script = tmp_path / "run_with_bg_dup2.py"
    runner_script.write_text(
        f"""#!/usr/bin/env python3
import os, sys, threading, time
from agy_token_viz.pty_runner import PTYVisualizerRunner
from agy_token_viz.state import SessionTokenState

state = SessionTokenState()

def _simulate_llama_cpp_dup2():
    devnull = os.open(os.devnull, os.O_RDWR)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        time.sleep(0.25)
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)

runner = PTYVisualizerRunner(cmd=[{str(fake_agy)!r}], state=state)
rc = runner.run(
    on_before_spawn=lambda: threading.Thread(
        target=_simulate_llama_cpp_dup2, daemon=True
    ).start()
)
raise SystemExit(rc)
""",
        encoding="utf-8",
    )

    master_fd, slave_fd = pty.openpty()
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "TERM": "xterm-256color",
    }
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(runner_script)],
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
    assert b"\x1b[48;2;242;146;46m  \x1b[48;2;240;114;54m  " in captured
    assert b"Antigravity CLI 1.5.5" in captured
    assert b"user@example.com" in captured


def test_bash_shim_discovers_real_agy_through_symlinks_on_path(tmp_path: Path) -> None:
    fake_agy = _make_fake_agy_script(tmp_path)
    shim_link_dir = tmp_path / "shim_links"
    real_bin_dir = tmp_path / "real_bin"
    shim_link_dir.mkdir()
    real_bin_dir.mkdir()

    # Symlink to bin/agy first on PATH, real agy second on PATH
    shim_symlink = shim_link_dir / "agy"
    shim_symlink.symlink_to(BIN_AGY)
    real_symlink = real_bin_dir / "agy"
    real_symlink.symlink_to(fake_agy)

    env = {
        **os.environ,
        "PATH": f"{shim_link_dir}:{real_bin_dir}:{os.environ.get('PATH', '')}",
        "AGY_DATA_DIR": str(tmp_path / "agy_data"),
    }
    env.pop("AGY_REAL_BIN", None)

    proc = subprocess.run(
        [str(shim_symlink), "-p", "via symlink"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_AGY_ARGS:["-p", "via symlink"]' in proc.stdout


