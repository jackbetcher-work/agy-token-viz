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
    filter_footer_mouse_events,
    inject_scroll_region_guards,
    set_pty_winsize,
    split_complete_terminal_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIN_AGY = PROJECT_ROOT / "bin" / "agy"
BIN_JETSKI = PROJECT_ROOT / "bin" / "jetski"


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


def test_inject_scroll_region_guards_restores_margins_after_1049h_and_1049l() -> None:
    enter_payload = b"\x1b[?1049h\x1b[H\x1b[2JWelcome to Jetski\r\n"
    guarded_enter = inject_scroll_region_guards(enter_payload, child_rows=22)
    assert guarded_enter.startswith(
        b"\x1b[?1049h\x1b7\x1b[1;22r\x1b8\x1b[H\x1b[2JWelcome to Jetski"
    )

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


def _make_fake_jetski_script(tmp_path: Path) -> Path:
    fake_jetski = tmp_path / "fake_jetski"
    fake_jetski.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys, time, uuid

print("FAKE_JETSKI_ARGS:" + json.dumps(sys.argv[1:]), flush=True)
data_dir = os.environ.get("JETSKI_DATA_DIR")
if data_dir:
    conv_id = str(uuid.uuid4())
    logs = pathlib.Path(data_dir) / "brain" / conv_id / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 2500,
        "cache_read_tokens": 10000,
        "output_tokens": 400,
        "thinking": "Running blaze test suite",
        "tool_calls": [{"name": "run_command", "args": {"CommandLine": "blaze test //pkg:all"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")
    time.sleep(0.25)
""",
        encoding="utf-8",
    )
    fake_jetski.chmod(fake_jetski.stat().st_mode | stat.S_IXUSR)
    return fake_jetski


def test_jetski_bash_shim_passthrough_bypasses_and_visualize_tokens(tmp_path: Path) -> None:
    fake_jetski = _make_fake_jetski_script(tmp_path)
    # Also set AGY_DATA_DIR to a decoy path to verify jetski never reads AGY_DATA_DIR
    decoy_agy_dir = tmp_path / "decoy_agy"
    decoy_logs = decoy_agy_dir / "brain" / "decoy" / ".system_generated" / "logs"
    decoy_logs.mkdir(parents=True, exist_ok=True)
    (decoy_logs / "transcript.jsonl").write_text(
        '{"step_index":1,"type":"PLANNER_RESPONSE","status":"DONE","input_tokens":99999,"output_tokens":1111}\n',
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "JETSKI_REAL_BIN": str(fake_jetski),
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_data"),
        "AGY_DATA_DIR": str(decoy_agy_dir),
    }

    # 1. Zero-overhead passthrough without --visualize-tokens
    proc_plain = subprocess.run(
        [str(BIN_JETSKI), "-p", "quick check"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_JETSKI_ARGS:["-p", "quick check"]' in proc_plain.stdout
    assert "[jetski-token-viz]" not in proc_plain.stderr

    # 2. Non-session Jetski subcommands & --stamp bypass visualizer even with --visualize-tokens
    #    (including when aliased as `jetski --visualize-tokens --dangerously-skip-permissions`)
    for bypass_args in (
        ["--stamp"],
        ["web"],
        ["chatski"],
        ["swarm", "capsules"],
        ["--dangerously-skip-permissions", "models"],
        ["--dangerously-skip-permissions", "swarm", "capsules"],
    ):
        proc_bypass = subprocess.run(
            [str(BIN_JETSKI), "--visualize-tokens", *bypass_args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert f"FAKE_JETSKI_ARGS:{BypassJson.dumps(bypass_args)}" in proc_bypass.stdout if (BypassJson := __import__("json")) else True
        assert "[jetski-token-viz]" not in proc_bypass.stderr

    # 3. Full visualization run via `bin/jetski --visualize-tokens`
    proc_viz = subprocess.run(
        [
            str(BIN_JETSKI),
            "--visualize-tokens",
            "--viz-heuristic-only",
            "-p",
            "run tests",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_JETSKI_ARGS:["-p", "run tests"]' in proc_viz.stdout
    assert "[jetski-token-viz] Session Token Summary" in proc_viz.stderr
    assert "Testing" in proc_viz.stderr
    assert "2.9k" in proc_viz.stderr


def test_jetski_bash_shim_discovers_real_jetski_through_symlinks_on_path(
    tmp_path: Path,
) -> None:
    fake_jetski = _make_fake_jetski_script(tmp_path)
    shim_link_dir = tmp_path / "jetski_shim_links"
    real_bin_dir = tmp_path / "jetski_real_bin"
    shim_link_dir.mkdir()
    real_bin_dir.mkdir()

    shim_symlink = shim_link_dir / "jetski"
    shim_symlink.symlink_to(BIN_JETSKI)
    real_symlink = real_bin_dir / "jetski"
    real_symlink.symlink_to(fake_jetski)

    env = {
        **os.environ,
        "PATH": f"{shim_link_dir}:{real_bin_dir}:{os.environ.get('PATH', '')}",
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_data"),
    }
    env.pop("JETSKI_REAL_BIN", None)
    env.pop("TOKEN_VIZ_REAL_BIN", None)

    proc = subprocess.run(
        [str(shim_symlink), "-p", "jetski via symlink"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'FAKE_JETSKI_ARGS:["-p", "jetski via symlink"]' in proc.stdout


def test_filter_footer_mouse_events_clamps_wheel_drag_and_suppresses_footer_clicks() -> None:
    # Viewport rows: 1..22; reserved footer rows: 23..24
    # 1. Normal click inside viewport passes through untouched
    assert (
        filter_footer_mouse_events(b"\x1b[<0;15;10M\x1b[<0;15;10m", child_rows=22)
        == b"\x1b[<0;15;10M\x1b[<0;15;10m"
    )

    # 2. Scroll wheel over footer rows (btn 64/65) clamps row to child_rows (22)
    assert (
        filter_footer_mouse_events(b"\x1b[<64;40;24M\x1b[<65;40;23M", child_rows=22)
        == b"\x1b[<64;40;22M\x1b[<65;40;22M"
    )

    # 3. Fresh click + drag + release starting ON footer row 23 is suppressed completely
    state = [False]
    assert filter_footer_mouse_events(
        b"\x1b[<0;12;23M", child_rows=22, suppressing_click=state
    ) == b""
    assert state[0] is True
    assert filter_footer_mouse_events(
        b"\x1b[<32;14;24M\x1b[<0;14;24m", child_rows=22, suppressing_click=state
    ) == b""
    assert state[0] is False

    # 4. Selection drag starting INSIDE viewport (row 15) then dragged & released over footer (row 24)
    #    clamps motion and mouse-up to row 22 so TUI selection never gets stuck!
    drag_seq = b"\x1b[<0;10;15M\x1b[<32;10;24M\x1b[<0;10;24m"
    assert (
        filter_footer_mouse_events(drag_seq, child_rows=22)
        == b"\x1b[<0;10;15M\x1b[<32;10;22M\x1b[<0;10;22m"
    )


def test_pty_alt_screen_synchronized_output_renders_footer_and_restores_on_crash(
    tmp_path: Path,
) -> None:
    """
    Simulate Jetski's default full-session alt-screen TUI (`?1049h`) with Ultraviolet
    DEC Mode 2026 synchronized frames (`?2026h ... ?2026l`), 24-bit RGB backgrounds,
    and `\x1b[J` screen clears, verifying:
    - Child PTY receives `rows - 2` window height
    - Live 2-line footer is painted while inside alt-screen (`?1049h`)
    - Footer resets SGR (`\x1b7\x1b[0m`) before `\x1b[2K` and injects before `\x1b[?2026l`
    - Abnormal exit without `\x1b[?1049l` automatically restores normal screen in `finally:`
    """
    fake_jetski_tui = tmp_path / "fake_jetski_tui"
    fake_jetski_tui.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, struct, sys, termios, time

packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
rows, cols, _, _ = struct.unpack("HHHH", packed)

# Enter alt-screen for the entire session (no ?1049l emitted before exit to test crash recovery too)
sys.stdout.write("\\x1b[?1049h\\x1b[?1006h")
sys.stdout.write(
    f"\\x1b[?2026h\\x1b[48;2;24;24;37m\\x1b[H\\x1b[JCHILD_WINSIZE:{rows}x{cols}\\x1b[?2026l"
)
sys.stdout.flush()

data_dir = os.environ.get("JETSKI_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "alt-conv-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 4000,
        "cache_read_tokens": 0,
        "output_tokens": 350,
        "content": "Alt-screen response",
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

time.sleep(0.35)
""",
        encoding="utf-8",
    )
    fake_jetski_tui.chmod(fake_jetski_tui.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 24, 100)
    env = {
        **os.environ,
        "JETSKI_REAL_BIN": str(fake_jetski_tui),
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_alt_data"),
        "TERM": "xterm-256color",
    }
    proc = subprocess.Popen(
        [str(BIN_JETSKI), "--visualize-tokens", "--viz-heuristic-only"],
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

    raw = bytes(captured)
    text = raw.decode("utf-8", errors="replace")
    assert "CHILD_WINSIZE:22x100" in text

    # Between ?1049h and the abnormal-exit ?1049l recovery, the footer must be painted on rows 23 & 24
    alt_start = raw.index(b"\x1b[?1049h")
    alt_end = raw.rindex(b"\x1b[?1049l")
    during_alt = raw[alt_start:alt_end]
    assert b"\x1b7\x1b[0m" in during_alt
    assert b"\x1b[23;1H\x1b[2K" in during_alt
    assert b"\x1b[24;1H\x1b[2K" in during_alt
    assert b"\x1b8\x1b[?2026l" in during_alt
    assert "[jetski-token-viz] Session Token Summary" in text



