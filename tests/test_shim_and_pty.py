"""Integration tests for `bin/agy` Bash shim and PTY wrapper (`pty_runner.py`)."""

from __future__ import annotations

import os
from pathlib import Path
import pty
import select
import signal
import stat
import subprocess
import time

from agy_token_viz.pty_runner import (
    PTYVisualizerRunner,
    build_resize_transition_sequence,
    clamp_child_cursor_positions,
    erase_normal_footer_before_alt_screen,
    filter_footer_mouse_events,
    inject_scroll_region_guards,
    rewrite_terminal_input_events,
    set_pty_winsize,
    split_complete_terminal_bytes,
)
from agy_token_viz.state import SessionTokenState


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


def test_inject_scroll_region_guards_preserves_subregions_and_clamps_resets() -> None:
    """
    Regression test: sub-viewport DECSTBM scroll regions (e.g. `\\x1b[1;22r` used by
    Jetski/Ultraviolet `scrolln` above the sticky user prompt) must NOT be overwritten
    with `1..child_rows` before the scroll command runs, whereas margin resets (`\\x1b[r`,
    `\\x1b[;r`, `0` bottom) and out-of-bounds bottom margins (`> child_rows`) must clamp
    inline to `child_rows`.
    """
    active: list[tuple[int, int] | None] = [None]

    # 1. Sub-viewport scroll region stays untouched and records active sub-region
    assert (
        inject_scroll_region_guards(b"\x1b[1;22r", child_rows=28, active_region=active)
        == b"\x1b[1;22r"
    )
    assert active[0] == (1, 22)

    assert (
        inject_scroll_region_guards(b"\x1b[5;22r", child_rows=28, active_region=active)
        == b"\x1b[5;22r"
    )
    assert active[0] == (5, 22)

    # 2. Full Ultraviolet `scrolln` frame: sub-region preserved during `\x1b[2S`, reset `\x1b[;r` clamped
    uv_scroll_frame = (
        b"\x1b[?2026h\x1b[1;22r\x1b[22;1H\x1b[2S\x1b[;r\x1b[21;1HNew line\x1b[?2026l"
    )
    assert inject_scroll_region_guards(
        uv_scroll_frame, child_rows=28, active_region=active
    ) == (
        b"\x1b[?2026h\x1b[1;22r\x1b[22;1H\x1b[2S\x1b[1;28r\x1b[21;1HNew line\x1b[?2026l"
    )
    assert active[0] is None

    # 3. Reset variants and out-of-bounds bottom margins clamp to child_rows
    for reset_seq in (b"\x1b[r", b"\x1b[;r", b"\x1b[0r", b"\x1b[0;0r", b"\x1b[1;30r", b"\x1b[28;30r"):
        assert (
            inject_scroll_region_guards(reset_seq, child_rows=28, active_region=active)
            == b"\x1b[1;28r"
        )
        assert active[0] is None

    # Top > 1 with omitted or out-of-bounds bottom preserves top and clamps bottom
    for partial_seq in (b"\x1b[5r", b"\x1b[5;r", b"\x1b[5;0r", b"\x1b[5;30r"):
        assert (
            inject_scroll_region_guards(partial_seq, child_rows=28, active_region=active)
            == b"\x1b[5;28r"
        )
        assert active[0] == (5, 28)

    # 4. Non-DECSTBM sequences ending in 'r' (`XTRESTORE`, `DECCARA`) pass through untouched
    non_decstbm = b"\x1b[?1006r\x1b[1;1;10;10$r"
    assert inject_scroll_region_guards(non_decstbm, child_rows=28) == non_decstbm

    # 5. Footer repaint preserves active sub-region (or clamps it to child_rows) if chunk splits mid-scroll
    runner = PTYVisualizerRunner(cmd=["true"], state=SessionTokenState(), use_color=False)
    assert b"\x1b[1;22r" in runner._build_footer_sequence(
        rows=30, cols=80, reapply_scroll_region=True, scroll_region=(1, 22)
    )
    assert b"\x1b[5;28r" in runner._build_footer_sequence(
        rows=30, cols=80, reapply_scroll_region=True, scroll_region=(5, 30)
    )
    assert b"\x1b[1;28r" in runner._build_footer_sequence(
        rows=30, cols=80, reapply_scroll_region=True, scroll_region=None
    )


class _MiniVT100:
    """Minimal VT100/xterm screen emulator for verifying sub-region scrolling and footer rows."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.norm_grid = self._blank_grid()
        self.alt_grid = self._blank_grid()
        self.grid = self.norm_grid
        self.last_alt_snapshot: list[str] = []
        self.row = 0
        self.col = 0
        self.pending_wrap = False
        self.scroll_top = 0
        self.scroll_bottom = rows - 1
        self._saved_cursor = (0, 0, False)
        self._alt_saved_cursor = (0, 0, False)

    def _blank_grid(self) -> list[list[str]]:
        return [[" "] * self.cols for _ in range(self.rows)]

    def snapshot_lines(self) -> list[str]:
        return ["".join(r).rstrip() for r in self.grid]

    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize terminal grid like xterm/Terminal.app/iTerm2 (resets DECSTBM to full screen)."""
        in_alt = self.grid is self.alt_grid

        def _resize_grid(old: list[list[str]]) -> list[list[str]]:
            out = [[" "] * new_cols for _ in range(new_rows)]
            copy_r = min(len(old), new_rows)
            for r in range(copy_r):
                copy_c = min(len(old[r]), new_cols)
                out[r][:copy_c] = old[r][:copy_c]
            return out

        self.norm_grid = _resize_grid(self.norm_grid)
        self.alt_grid = _resize_grid(self.alt_grid)
        self.grid = self.alt_grid if in_alt else self.norm_grid
        self.rows = new_rows
        self.cols = new_cols
        self.row = max(0, min(new_rows - 1, self.row))
        self.col = max(0, min(new_cols - 1, self.col))
        self.pending_wrap = False
        # Real terminal emulators reset DECSTBM margins to 1..new_rows on window resize
        self.scroll_top = 0
        self.scroll_bottom = new_rows - 1
        sr, sc, _ = self._saved_cursor
        self._saved_cursor = (
            max(0, min(new_rows - 1, sr)),
            max(0, min(new_cols - 1, sc)),
            False,
        )
        ar, ac, _ = self._alt_saved_cursor
        self._alt_saved_cursor = (
            max(0, min(new_rows - 1, ar)),
            max(0, min(new_cols - 1, ac)),
            False,
        )

    def _scroll_up(self, count: int) -> None:
        span = self.scroll_bottom - self.scroll_top + 1
        n = max(0, min(count, span))
        for r in range(self.scroll_top, self.scroll_bottom - n + 1):
            self.grid[r] = self.grid[r + n][:]
        for r in range(self.scroll_bottom - n + 1, self.scroll_bottom + 1):
            self.grid[r] = [" "] * self.cols

    def _scroll_down(self, count: int) -> None:
        span = self.scroll_bottom - self.scroll_top + 1
        n = max(0, min(count, span))
        for r in range(self.scroll_bottom, self.scroll_top + n - 1, -1):
            self.grid[r] = self.grid[r - n][:]
        for r in range(self.scroll_top, self.scroll_top + n):
            self.grid[r] = [" "] * self.cols

    def feed(self, raw: bytes) -> None:
        s = raw.decode("utf-8", errors="replace")
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "\x1b":
                if i + 1 >= n:
                    break
                nxt = s[i + 1]
                if nxt == "7":
                    self._saved_cursor = (self.row, self.col, self.pending_wrap)
                    i += 2
                    continue
                if nxt == "8":
                    self.row, self.col, self.pending_wrap = self._saved_cursor
                    i += 2
                    continue
                if nxt == "D":
                    self.pending_wrap = False
                    if self.row == self.scroll_bottom:
                        self._scroll_up(1)
                    elif self.row < self.rows - 1:
                        self.row += 1
                    i += 2
                    continue
                if nxt == "M":
                    self.pending_wrap = False
                    if self.row == self.scroll_top:
                        self._scroll_down(1)
                    elif self.row > 0:
                        self.row -= 1
                    i += 2
                    continue
                if nxt == "[":
                    j = i + 2
                    while j < n and not (0x40 <= ord(s[j]) <= 0x7E):
                        j += 1
                    if j >= n:
                        break
                    body = s[i + 2 : j]
                    final = s[j]
                    self._handle_csi(body, final)
                    i = j + 1
                    continue
                i += 2
                continue
            if ch == "\r":
                self.col = 0
                self.pending_wrap = False
            elif ch == "\n":
                self.pending_wrap = False
                if self.row == self.scroll_bottom:
                    self._scroll_up(1)
                elif self.row < self.rows - 1:
                    self.row += 1
            elif ord(ch) >= 0x20:
                if self.pending_wrap:
                    self.col = 0
                    self.pending_wrap = False
                    if self.row == self.scroll_bottom:
                        self._scroll_up(1)
                    elif self.row < self.rows - 1:
                        self.row += 1
                self.grid[self.row][self.col] = ch
                if self.col + 1 >= self.cols:
                    self.pending_wrap = True
                else:
                    self.col += 1
            i += 1

    def _handle_csi(self, body: str, final: str) -> None:
        if body.startswith("?"):
            modes = [m for m in body[1:].split(";") if m]
            if final == "h" and any(m in ("1049", "1047", "47") for m in modes):
                self._alt_saved_cursor = (self.row, self.col, self.pending_wrap)
                self.alt_grid = self._blank_grid()
                self.grid = self.alt_grid
                self.row, self.col, self.pending_wrap = 0, 0, False
                self.scroll_top, self.scroll_bottom = 0, self.rows - 1
            elif final == "l" and any(m in ("1049", "1047", "47") for m in modes):
                self.last_alt_snapshot = ["".join(r).rstrip() for r in self.alt_grid]
                self.grid = self.norm_grid
                self.row, self.col, self.pending_wrap = self._alt_saved_cursor
                self.scroll_top, self.scroll_bottom = 0, self.rows - 1
            return

        if any(not (c.isdigit() or c == ";") for c in body):
            return
        parts = body.split(";") if body else []
        p0 = int(parts[0]) if len(parts) >= 1 and parts[0] else 0
        p1 = int(parts[1]) if len(parts) >= 2 and parts[1] else 0

        if final in ("H", "f"):
            self.row = max(0, min(self.rows - 1, (p0 or 1) - 1))
            self.col = max(0, min(self.cols - 1, (p1 or 1) - 1))
            self.pending_wrap = False
        elif final == "d":
            self.row = max(0, min(self.rows - 1, (p0 or 1) - 1))
            self.pending_wrap = False
        elif final == "G":
            self.col = max(0, min(self.cols - 1, (p0 or 1) - 1))
            self.pending_wrap = False
        elif final == "A":
            self.row = max(0, self.row - (p0 or 1))
            self.pending_wrap = False
        elif final == "B":
            self.row = min(self.rows - 1, self.row + (p0 or 1))
            self.pending_wrap = False
        elif final == "C":
            self.col = min(self.cols - 1, self.col + (p0 or 1))
            self.pending_wrap = False
        elif final == "D":
            self.col = max(0, self.col - (p0 or 1))
            self.pending_wrap = False
        elif final == "J":
            self.pending_wrap = False
            if p0 == 2:
                for r in range(self.rows):
                    self.grid[r] = [" "] * self.cols
            elif p0 == 0:
                for c in range(self.col, self.cols):
                    self.grid[self.row][c] = " "
                for r in range(self.row + 1, self.rows):
                    self.grid[r] = [" "] * self.cols
        elif final == "K":
            self.pending_wrap = False
            if p0 == 2:
                self.grid[self.row] = [" "] * self.cols
            elif p0 == 0:
                for c in range(self.col, self.cols):
                    self.grid[self.row][c] = " "
        elif final == "r":
            top = p0 or 1
            bot = (p1 if len(parts) >= 2 and parts[1] else 0) or self.rows
            if 1 <= top < bot <= self.rows:
                self.scroll_top = top - 1
                self.scroll_bottom = bot - 1
                self.row, self.col, self.pending_wrap = 0, 0, False
        elif final == "S":
            self.pending_wrap = False
            self._scroll_up(p0 or 1)
        elif final == "T":
            self.pending_wrap = False
            self._scroll_down(p0 or 1)


def test_jetski_subregion_hardware_scroll_never_overlaps_prompt_or_footer(
    tmp_path: Path,
) -> None:
    """
    End-to-end regression test reproducing the Jetski `--visualize-tokens` UI overlap bug:
    - Outer terminal is 30x80; child PTY gets 28x80 (rows 1..22 = conversation viewport,
      rows 23..28 = sticky user prompt box, rows 29..30 = live 2-line token footer).
    - Fake Jetski TUI paints the sticky user prompt box ONCE on frame 1 (matching
      Ultraviolet's diff renderer `curbuf`, which assumes rows 23..28 never move when
      scrolling inside `\\x1b[1;22r`) and then streams generated text using Ultraviolet's
      exact hardware scroll sequences:
        1. `\\x1b[1;22r` + `\\x1b[1;1H\\x1b[2T` + `\\x1b[;r` (multi-line scroll down)
        2. `\\x1b[1;22r` + `\\x1b[22;1H\\x1b[3S` + `\\x1b[;r` (multi-line scroll up)
        3. `\\x1b[1;22r` + `\\x1b[22;1H\\n` + `\\x1b[;r` (single-line LF scroll up)
    - Verifies on a VT100 screen emulator that generated text never overlaps the user
      prompt area, the user prompt box never scrolls up into the generated text area,
      and the 2-line token footer stays intact on rows 29..30.
    """
    fake_jetski_scroll = tmp_path / "fake_jetski_scroll"
    fake_jetski_scroll.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys, time

data_dir = os.environ.get("JETSKI_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "scroll-conv-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 3200,
        "cache_read_tokens": 6400,
        "output_tokens": 450,
        "content": "Streaming response",
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

# Frame 1: Initial full render (rows 1..22 = generated text, rows 23..28 = sticky user prompt)
out = ["\\x1b[?1049h\\x1b[?2026h\\x1b[H\\x1b[2J"]
for r in range(1, 23):
    out.append(f"\\x1b[{r};1HGen line {r:02d} - initial response content")
prompt_rows = [
    "┌────────────────────── User Prompt ──────────────────────┐",
    "│ > Follow-up prompt typed here                           │",
    "│                                                         │",
    "│ [Enter to send]                                         │",
    "└─────────────────────────────────────────────────────────┘",
    "  gemini-2.5-pro · /Users/dev/project",
]
for idx, p_line in enumerate(prompt_rows, start=23):
    out.append(f"\\x1b[{idx};1H{p_line}")
out.append("\\x1b[?2026l")
sys.stdout.write("".join(out))
sys.stdout.flush()
time.sleep(0.08)

# Frame 2: Sub-region scroll DOWN by 2 (`scrollDown`: `\\x1b[1;22r` + `\\x1b[1;1H\\x1b[2T` + `\\x1b[;r`)
# Must clip at row 22 and NEVER push `Gen line 21/22` down into prompt rows 23..24!
sys.stdout.write(
    "\\x1b[?2026h"
    "\\x1b[1;22r\\x1b[1;1H\\x1b[2T\\x1b[;r"
    "\\x1b[1;1HInserted top 01"
    "\\x1b[2;1HInserted top 02"
    "\\x1b[?2026l"
)
sys.stdout.flush()
time.sleep(0.08)

# Frame 3: Sub-region scroll UP by 3 (`scrollUp`: `\\x1b[1;22r` + `\\x1b[22;1H\\x1b[3S` + `\\x1b[;r`)
# Must NOT pull prompt rows 23..25 up into rows 20..22! Diff-paint without `\\x1b[2K`.
sys.stdout.write(
    "\\x1b[?2026h"
    "\\x1b[1;22r\\x1b[22;1H\\x1b[3S\\x1b[;r"
    "\\x1b[20;1HGen tail A"
    "\\x1b[21;1HGen tail B"
    "\\x1b[22;1HGen tail C - long trailing words"
    "\\x1b[?2026l"
)
sys.stdout.flush()
time.sleep(0.08)

# Frame 4: Sub-region 1-line LF scroll UP (`\\x1b[1;22r` + `\\x1b[22;1H\\n` + `\\x1b[;r`)
# Row 22 must scroll `Gen tail C - long trailing words` up to row 21 and blank row 22
# so short `FINAL_END` has no trailing leftover characters!
sys.stdout.write(
    "\\x1b[?2026h"
    "\\x1b[1;22r\\x1b[22;1H\\n\\x1b[;r"
    "\\x1b[22;1HFINAL_END"
    "\\x1b[?2026l"
)
sys.stdout.flush()
time.sleep(0.25)

# Cleanly exit alt-screen so `_MiniVT100` snapshots the final alt-screen grid
sys.stdout.write("\\x1b[?1049l")
sys.stdout.flush()
""",
        encoding="utf-8",
    )
    fake_jetski_scroll.chmod(fake_jetski_scroll.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 30, 80)
    env = {
        **os.environ,
        "JETSKI_REAL_BIN": str(fake_jetski_scroll),
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_scroll_data"),
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

    vt = _MiniVT100(rows=30, cols=80)
    vt.feed(bytes(captured))
    snap = vt.last_alt_snapshot
    assert len(snap) == 30

    # 1. Conversation viewport (rows 1..22, 0-indexed 0..21): exact scrolled generated text,
    #    zero prompt box characters (`┌`, `│`, `└`, `User Prompt`), zero un-cleared tail bleed.
    assert snap[18] == "Gen tail A"
    assert snap[19] == "Gen tail B"
    assert snap[20] == "Gen tail C - long trailing words"
    assert snap[21] == "FINAL_END"
    for r in range(0, 22):
        assert "User Prompt" not in snap[r]
        assert "Follow-up prompt" not in snap[r]
        assert "┌" not in snap[r] and "└" not in snap[r]

    # 2. Sticky user prompt area (rows 23..28, 0-indexed 22..27): 100% intact, zero generated text overlap
    assert snap[22] == "┌────────────────────── User Prompt ──────────────────────┐"
    assert snap[23] == "│ > Follow-up prompt typed here                           │"
    assert snap[24] == "│                                                         │"
    assert snap[25] == "│ [Enter to send]                                         │"
    assert snap[26] == "└─────────────────────────────────────────────────────────┘"
    assert snap[27] == "  gemini-2.5-pro · /Users/dev/project"

    # 3. Reserved 2-line token visualizer footer (rows 29..30, 0-indexed 28..29): intact at bottom
    assert "Tokens" in snap[28]
    assert "In:" in snap[28] or "I:" in snap[28]
    assert snap[29] != ""


def test_build_resize_transition_sequence_clears_stale_and_newly_exposed_rows() -> None:
    # 1. Expanding (24 -> 54): prev_child_rows = 22, new_child_rows = 52 -> clear from row 23 down
    assert (
        build_resize_transition_sequence(
            prev_rows=24,
            prev_use_footer=True,
            new_rows=54,
            new_use_footer=True,
        )
        == b"\x1b7\x1b[0m\x1b[1;52r\x1b[23;1H\x1b[J\x1b8"
    )

    # 2. Shrinking (54 -> 24): prev_child_rows = 52, new_child_rows = 22 -> clear from row 23 down
    assert (
        build_resize_transition_sequence(
            prev_rows=54,
            prev_use_footer=True,
            new_rows=24,
            new_use_footer=True,
        )
        == b"\x1b7\x1b[0m\x1b[1;22r\x1b[23;1H\x1b[J\x1b8"
    )

    # 3. Width-only resize (54 -> 54): clear reserved footer band 53..54
    assert (
        build_resize_transition_sequence(
            prev_rows=54,
            prev_use_footer=True,
            new_rows=54,
            new_use_footer=True,
        )
        == b"\x1b7\x1b[0m\x1b[1;52r\x1b[53;1H\x1b[J\x1b8"
    )

    # 4. Shrinking across MIN_TERMINAL_ROWS_FOR_FOOTER (6 -> 5): prev_child_rows = 4 -> clear row 5 & reset margins
    assert (
        build_resize_transition_sequence(
            prev_rows=6,
            prev_use_footer=True,
            new_rows=5,
            new_use_footer=False,
        )
        == b"\x1b7\x1b[0m\x1b[r\x1b[5;1H\x1b[J\x1b8"
    )

    # 5. Shrinking from 24 -> 4: old footer at 23..24 was clipped; reset margins without erasing 1..4
    assert (
        build_resize_transition_sequence(
            prev_rows=24,
            prev_use_footer=True,
            new_rows=4,
            new_use_footer=False,
        )
        == b"\x1b7\x1b[0m\x1b[r\x1b8"
    )

    # 6. Expanding from < 6 rows to >= 6 rows (4 -> 24): clear newly exposed 5..24 & set 1..22 margins
    assert (
        build_resize_transition_sequence(
            prev_rows=4,
            prev_use_footer=False,
            new_rows=24,
            new_use_footer=True,
        )
        == b"\x1b7\x1b[0m\x1b[1;22r\x1b[5;1H\x1b[J\x1b8"
    )

    # 7. Tiny -> tiny (4 -> 5): no-op
    assert (
        build_resize_transition_sequence(
            prev_rows=4,
            prev_use_footer=False,
            new_rows=5,
            new_use_footer=False,
        )
        == b""
    )


def test_double_click_fullscreen_resize_animation_never_duplicates_bars_or_blocks_prompt(
    tmp_path: Path,
) -> None:
    """
    End-to-end regression test reproducing the macOS double-click terminal full-screen bug:
    - Child TUI uses an Ultraviolet-style diff renderer (`curbuf`) that resizes `curbuf` by
      padding newly exposed rows/cols with `' '` and ONLY emits non-matching cells (never
      emitting `\\x1b[2J` on resize, so any stale token bar cells inside `1..child_rows` or
      inside the prompt box's blank interior would remain permanently visible).
    - Simulates:
      1. Initial `24x80` render (`child_rows=22`, prompt box on rows `17..22`, footer on `23..24`).
      2. Rapid 10-step double-click full-screen zoom animation (`24x80 -> ... -> 54x140`).
      3. Post-settle incremental expand (`54x140 -> 60x150`) where the new prompt box (`53..58`)
         lands directly on the exact rows (`53..54`) previously occupied by the settled footer.
      4. Rapid double-click restore/shrink animation (`60x150 -> ... -> 24x80`).
    """
    fake_diff_tui = tmp_path / "fake_diff_tui"
    fake_diff_tui.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, select, signal, struct, sys, termios

data_dir = os.environ.get("JETSKI_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "resize-conv-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 3500,
        "cache_read_tokens": 9000,
        "output_tokens": 600,
        "thinking": "Inspecting workspace files",
        "tool_calls": [{"name": "read_file", "args": {"Path": "README.md"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

wake_r, wake_w = os.pipe()
os.set_blocking(wake_r, False)
os.set_blocking(wake_w, False)
resized = [True]

def _on_winch(_s, _f):
    resized[0] = True
    try:
        os.write(wake_w, b"W")
    except OSError:
        pass

signal.signal(signal.SIGWINCH, _on_winch)

def _winsize():
    packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
    r, c, _, _ = struct.unpack("HHHH", packed)
    return max(6, r), max(20, c)

curbuf = []

def _resize_curbuf(rows, cols):
    global curbuf
    new_cur = [[" "] * cols for _ in range(rows)]
    for r in range(min(len(curbuf), rows)):
        copy_c = min(len(curbuf[r]), cols)
        new_cur[r][:copy_c] = curbuf[r][:copy_c]
    curbuf = new_cur

def _render_frame():
    rows, cols = _winsize()
    _resize_curbuf(rows, cols)
    nextbuf = [[" "] * cols for _ in range(rows)]

    def _put_line(r_idx, text):
        if 0 <= r_idx < rows:
            for c_idx, ch in enumerate(text[:cols]):
                nextbuf[r_idx][c_idx] = ch

    _put_line(0, f"READY_{rows}x{cols} - Antigravity / Jetski TUI")
    _put_line(1, "Assistant: Ready to help with your project.")

    box_w = max(20, cols - 2)
    top_border = "┌" + ("─" * (box_w - 2)) + "┐"
    prompt_txt = "> Type your prompt here..."
    prompt_line = "│ " + prompt_txt + (" " * max(0, box_w - 3 - len(prompt_txt))) + "│"
    empty_line = "│" + (" " * (box_w - 2)) + "│"
    hint_txt = "[Enter to send]"
    hint_line = "│ " + hint_txt + (" " * max(0, box_w - 3 - len(hint_txt))) + "│"
    bot_border = "└" + ("─" * (box_w - 2)) + "┘"
    status_line = f"  gemini-2.5-pro · {rows}x{cols}"

    prompt_block = [top_border, prompt_line, empty_line, hint_line, bot_border, status_line]
    start_row = max(2, rows - len(prompt_block))
    for offset, line in enumerate(prompt_block):
        _put_line(start_row + offset, line)

    # Diff-flush ONLY cells that changed relative to `curbuf` (no \\x1b[2J on resize!)
    out = ["\\x1b[?2026h"]
    for r in range(rows):
        c = 0
        while c < cols:
            if nextbuf[r][c] != curbuf[r][c]:
                start_c = c
                chars = []
                while c < cols and nextbuf[r][c] != curbuf[r][c]:
                    chars.append(nextbuf[r][c])
                    curbuf[r][c] = nextbuf[r][c]
                    c += 1
                out.append(f"\\x1b[{r + 1};{start_c + 1}H" + "".join(chars))
            else:
                c += 1
    out.append("\\x1b[?2026l")
    sys.stdout.write("".join(out))
    sys.stdout.flush()

sys.stdout.write("\\x1b[?1049h\\x1b[H\\x1b[2J")
sys.stdout.flush()

while True:
    if resized[0]:
        resized[0] = False
        _render_frame()
    try:
        rlist, _, _ = select.select([sys.stdin.fileno(), wake_r], [], [], 0.2)
    except InterruptedError:
        continue
    if wake_r in rlist:
        try:
            os.read(wake_r, 1024)
        except OSError:
            pass
    if sys.stdin.fileno() in rlist:
        data = os.read(sys.stdin.fileno(), 64)
        if not data or b"q" in data:
            break

sys.stdout.write("\\x1b[?1049l")
sys.stdout.flush()
""",
        encoding="utf-8",
    )
    fake_diff_tui.chmod(fake_diff_tui.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 24, 80)
    vt = _MiniVT100(rows=24, cols=80)

    env = {
        **os.environ,
        "JETSKI_REAL_BIN": str(fake_diff_tui),
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_resize_data"),
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

    def _drain_into_vt(duration: float) -> None:
        end_at = time.monotonic() + duration
        while True:
            rem = end_at - time.monotonic()
            if rem <= 0:
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.02, rem))
            if ready:
                try:
                    chunk = os.read(master_fd, 16384)
                    if chunk:
                        vt.feed(chunk)
                except OSError:
                    break

    def _wait_for_settled_screen(expected_child_rows: int, expected_cols: int) -> list[str]:
        marker = f"READY_{expected_child_rows}x{expected_cols}"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain_into_vt(0.03)
            lines = vt.snapshot_lines()
            if (
                len(lines) == expected_child_rows + 2
                and marker in lines[0]
                and "Tokens" in lines[expected_child_rows]
                and "Explore" in lines[expected_child_rows + 1]
            ):
                return lines
        raise AssertionError(
            f"Timed out waiting for {marker} + settled footer; screen:\n"
            + "\n".join(f"{i + 1:02d}: {l}" for i, l in enumerate(vt.snapshot_lines()))
        )

    def _apply_resize_step(new_rows: int, new_cols: int) -> None:
        vt.resize(new_rows, new_cols)
        set_pty_winsize(master_fd, new_rows, new_cols)
        os.kill(proc.pid, signal.SIGWINCH)

    def _assert_clean_layout(lines: list[str], total_rows: int) -> None:
        child_rows = total_rows - 2
        prompt_start = child_rows - 6
        # 1. Zero duplicate/ghost token bars anywhere in child viewport OR inside prompt box!
        for r in range(child_rows):
            assert "Tokens" not in lines[r], f"Ghost token bar on row {r + 1}: {lines[r]!r}"
            assert "Explore" not in lines[r], f"Ghost category line on row {r + 1}: {lines[r]!r}"
            assert "Cache:" not in lines[r], f"Ghost bar stats on row {r + 1}: {lines[r]!r}"
            assert "█" not in lines[r] and "▓" not in lines[r], (
                f"Ghost bar block glyph on row {r + 1}: {lines[r]!r}"
            )

        # 2. Prompt box at bottom of child_rows is 100% intact and unblocked
        assert lines[prompt_start].startswith("┌─") and lines[prompt_start].endswith("┐")
        assert lines[prompt_start + 1].startswith("│ > Type your prompt here...")
        assert lines[prompt_start + 1].endswith("│")
        assert lines[prompt_start + 2].startswith("│ ") and lines[prompt_start + 2].endswith("│")
        assert lines[prompt_start + 2][1:-1].strip() == ""
        assert "[Enter to send]" in lines[prompt_start + 3]
        assert lines[prompt_start + 4].startswith("└─") and lines[prompt_start + 4].endswith("┘")
        assert "gemini-2.5-pro" in lines[prompt_start + 5]

        # 3. Exactly one 2-line token footer on the bottom 2 rows
        assert "Tokens" in lines[total_rows - 2]
        assert "Explore" in lines[total_rows - 1]

    try:
        # Phase 1: Initial 24x80 screen
        snap_init = _wait_for_settled_screen(22, 80)
        _assert_clean_layout(snap_init, 24)

        # Phase 2: Double-click titlebar -> rapid 10-step full-screen zoom animation (24x80 -> 54x140)
        zoom_steps = [
            (26, 85),
            (29, 92),
            (32, 100),
            (36, 108),
            (40, 116),
            (44, 124),
            (48, 130),
            (51, 135),
            (53, 138),
            (54, 140),
        ]
        for r_step, c_step in zoom_steps:
            _apply_resize_step(r_step, c_step)
            _drain_into_vt(0.012)

        snap_fullscreen = _wait_for_settled_screen(52, 140)
        _assert_clean_layout(snap_fullscreen, 54)

        # Phase 3: Post-settle expand (54x140 -> 60x150) where new prompt box (53..58) lands
        # directly on rows 53..54 previously occupied by the settled footer
        _apply_resize_step(60, 150)
        snap_expanded = _wait_for_settled_screen(58, 150)
        _assert_clean_layout(snap_expanded, 60)

        # Phase 4: Double-click titlebar again -> un-fullscreen animation back to 24x80
        unzoom_steps = [
            (52, 132),
            (44, 118),
            (36, 102),
            (29, 90),
            (24, 80),
        ]
        for r_step, c_step in unzoom_steps:
            _apply_resize_step(r_step, c_step)
            _drain_into_vt(0.012)

        snap_restored = _wait_for_settled_screen(22, 80)
        _assert_clean_layout(snap_restored, 24)

        os.write(master_fd, b"q\n")
        exit_deadline = time.monotonic() + 5.0
        while proc.poll() is None and time.monotonic() < exit_deadline:
            _drain_into_vt(0.05)
        assert proc.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_normal_buffer_inline_tui_fullscreen_resize_never_duplicates_token_bars(
    tmp_path: Path,
) -> None:
    """
    Verify inline normal-screen mode (`agy` after `?1049l`) across an animated full-screen
    resize (`24x80 -> ... -> 48x130`) leaves zero duplicate token bars above the footer.
    """
    fake_inline_agy = tmp_path / "fake_inline_agy"
    fake_inline_agy.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, select, signal, struct, sys, termios

data_dir = os.environ.get("AGY_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "inline-resize-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 2200,
        "cache_read_tokens": 4000,
        "output_tokens": 300,
        "thinking": "Editing file",
        "tool_calls": [{"name": "edit_file", "args": {"Path": "src/app.py"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

wake_r, wake_w = os.pipe()
os.set_blocking(wake_r, False)
os.set_blocking(wake_w, False)
resized = [True]

def _on_winch(_s, _f):
    resized[0] = True
    try:
        os.write(wake_w, b"W")
    except OSError:
        pass

signal.signal(signal.SIGWINCH, _on_winch)

def _winsize():
    packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
    r, c, _, _ = struct.unpack("HHHH", packed)
    return max(6, r), max(20, c)

# Simulate agy splash (?1049h) -> normal buffer (?1049l) inline prompt box
sys.stdout.write("\\x1b[?1049h\\x1b[H\\x1b[2JSigning in...\\x1b[?1049l\\r\\x1b[J")
sys.stdout.flush()

prev_lines = 0
while True:
    if resized[0]:
        resized[0] = False
        rows, cols = _winsize()
        if prev_lines > 1:
            sys.stdout.write(f"\\x1b[{prev_lines - 1}A")
        box_w = max(20, min(cols - 2, 70))
        block = [
            f"INLINE_READY_{rows}x{cols}",
            "┌" + ("─" * (box_w - 2)) + "┐",
            "│ > Ask Antigravity..." + (" " * max(0, box_w - 23)) + "│",
            "└" + ("─" * (box_w - 2)) + "┘",
        ]
        sys.stdout.write("\\r\\x1b[J" + "\\r\\n".join(block))
        sys.stdout.flush()
        prev_lines = len(block)

    try:
        rlist, _, _ = select.select([sys.stdin.fileno(), wake_r], [], [], 0.2)
    except InterruptedError:
        continue
    if wake_r in rlist:
        try:
            os.read(wake_r, 1024)
        except OSError:
            pass
    if sys.stdin.fileno() in rlist:
        data = os.read(sys.stdin.fileno(), 64)
        if not data or b"q" in data:
            break
""",
        encoding="utf-8",
    )
    fake_inline_agy.chmod(fake_inline_agy.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 24, 80)
    vt = _MiniVT100(rows=24, cols=80)

    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_inline_agy),
        "AGY_DATA_DIR": str(tmp_path / "agy_inline_resize_data"),
        "TERM": "xterm-256color",
    }
    proc = subprocess.Popen(
        [str(BIN_AGY), "--visualize-tokens", "--viz-heuristic-only"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    def _drain(duration: float) -> None:
        end_at = time.monotonic() + duration
        while True:
            rem = end_at - time.monotonic()
            if rem <= 0:
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.02, rem))
            if ready:
                try:
                    chunk = os.read(master_fd, 16384)
                    if chunk:
                        vt.feed(chunk)
                except OSError:
                    break

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if any("INLINE_READY_22x80" in l for l in lines) and "Tokens" in lines[22]:
                break
        else:
            raise AssertionError("Timed out waiting for initial inline render")

        for r_step, c_step in [(28, 90), (33, 100), (39, 112), (44, 122), (48, 130)]:
            vt.resize(r_step, c_step)
            set_pty_winsize(master_fd, r_step, c_step)
            os.kill(proc.pid, signal.SIGWINCH)
            _drain(0.012)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if (
                len(lines) == 48
                and any("INLINE_READY_46x130" in l for l in lines[:46])
                and "Tokens" in lines[46]
                and "Code" in lines[47]
            ):
                break
        else:
            raise AssertionError(
                "Timed out waiting for settled 48x130 inline render:\n"
                + "\n".join(f"{i + 1:02d}: {l}" for i, l in enumerate(vt.snapshot_lines()))
            )

        lines = vt.snapshot_lines()
        for r in range(46):
            assert "Tokens" not in lines[r], f"Duplicate token bar on row {r + 1}: {lines[r]!r}"
        assert any("│ > Ask Antigravity..." in l for l in lines[:46])
        assert "Tokens" in lines[46]
        assert "Code" in lines[47]

        os.write(master_fd, b"q\n")
        exit_deadline = time.monotonic() + 5.0
        while proc.poll() is None and time.monotonic() < exit_deadline:
            _drain(0.05)
        assert proc.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_clamp_child_cursor_positions_prevents_footer_overwrites_and_size_probe_leaks() -> None:
    # 1. Valid child cursor targets (<= child_rows) and non-cursor CSI pass through untouched
    valid = b"\x1b[H\x1b[1;1H\x1b[22;80H\x1b[22d\x1b[38;2;255;100;50m\x1b[?25h"
    assert clamp_child_cursor_positions(valid, child_rows=22) == valid

    # 2. Out-of-bounds CUP (`H`), HVP (`f`), VPA (`d`), and `\x1b[999;999H\x1b[6n` size probes clamp to child_rows
    raw = (
        b"\x1b[23;1HFooterRow1"
        b"\x1b[24;15fFooterRow2"
        b"\x1b[30HColDefault"
        b"\x1b[28;fEmptyCol"
        b"\x1b[24dVPATarget"
        b"\x1b[999;999H\x1b[6n"
    )
    assert clamp_child_cursor_positions(raw, child_rows=22) == (
        b"\x1b[22;1HFooterRow1"
        b"\x1b[22;15fFooterRow2"
        b"\x1b[22HColDefault"
        b"\x1b[22;fEmptyCol"
        b"\x1b[22dVPATarget"
        b"\x1b[22;999H\x1b[6n"
    )


def test_rewrite_terminal_input_events_adjusts_xtwinops_inband_2048_and_cpr_replies() -> None:
    click_state = [False]
    # 1. XTWINOPS text-area size report (`\x1b[8;rows;colst`) -> child_rows
    assert (
        rewrite_terminal_input_events(
            b"\x1b[8;54;140t", child_rows=52, suppressing_click=click_state
        )
        == b"\x1b[8;52;140t"
    )

    # 2. DEC Mode 2048 in-band resize (`\x1b[48;rows;cols;px_h;px_wt`) -> child_rows & scaled pixel height
    assert (
        rewrite_terminal_input_events(
            b"\x1b[48;54;140;1080;2800t",
            child_rows=52,
            suppressing_click=click_state,
        )
        == b"\x1b[48;52;140;1040;2800t"
    )

    # 3. CPR cursor position reply (`\x1b[row;colR`) clamps > child_rows while preserving Shift+F3 (`\x1b[1;2R`)
    assert (
        rewrite_terminal_input_events(
            b"\x1b[54;80R\x1b[1;2R\x1b[20;10R",
            child_rows=52,
            suppressing_click=click_state,
        )
        == b"\x1b[52;80R\x1b[1;2R\x1b[20;10R"
    )

    # 4. Combined payload: footer click suppressed + in-band resize rewritten + normal keystrokes preserved
    combined = b"\x1b[<0;12;53M\x1b[<0;12;53m\x1b[48;54;100;540;1000thello"
    assert (
        rewrite_terminal_input_events(
            combined, child_rows=52, suppressing_click=click_state
        )
        == b"\x1b[48;52;100;520;1000thello"
    )


def test_erase_normal_footer_before_alt_screen_prevents_zombie_startup_bar_after_alt_resize(
    tmp_path: Path,
) -> None:
    """
    Verify both unit and end-to-end PTY behavior when launching at `24x80`, entering
    alt-screen (`?1049h`), resizing to `44x120`, and exiting alt-screen (`?1049l`) back
    to the normal screen buffer:
    - The startup normal-buffer token bar drawn at rows `23..24` is erased *before*
      `?1049h` saves the normal buffer, so rows `23..24` contain zero zombie bar remnants
      when `?1049l` restores the normal screen at `44x120`.
    """
    in_alt = [False]
    nf_state: list[tuple[int, int, int] | None] = [(24, 78, 70)]
    out = erase_normal_footer_before_alt_screen(
        b"pre\x1b[?1049hinside",
        in_alt_screen=in_alt,
        normal_footer_state=nf_state,
    )
    assert out == (
        b"pre\x1b7\x1b[0m\x1b[23;1H\x1b[2K\x1b[24;1H\x1b[2K\x1b[r\x1b8\x1b[?1049hinside"
    )
    assert in_alt[0] is True
    assert nf_state[0] is None

    fake_alt_then_inline = tmp_path / "fake_alt_then_inline"
    fake_alt_then_inline.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, select, signal, struct, sys, termios, time

data_dir = os.environ.get("JETSKI_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "alt-exit-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 3000,
        "cache_read_tokens": 5000,
        "output_tokens": 400,
        "tool_calls": [{"name": "read_file", "args": {"Path": "README.md"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

wake_r, wake_w = os.pipe()
os.set_blocking(wake_r, False)
os.set_blocking(wake_w, False)
resized = [True]

def _on_winch(_s, _f):
    resized[0] = True
    try:
        os.write(wake_w, b"W")
    except OSError:
        pass

signal.signal(signal.SIGWINCH, _on_winch)

def _winsize():
    packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
    r, c, _, _ = struct.unpack("HHHH", packed)
    return max(1, r), max(1, c)

# Print pre-alt banner on normal buffer, let startup footer settle, then enter ?1049h
sys.stdout.write("PRE_ALT_BANNER_LINE\\r\\n")
sys.stdout.flush()
time.sleep(0.12)

sys.stdout.write("\\x1b[?1049h")
sys.stdout.flush()

in_alt = True
while True:
    if resized[0]:
        resized[0] = False
        r, c = _winsize()
        if in_alt:
            sys.stdout.write(f"\\x1b[H\\x1b[2JALT_VIEW_{r}x{c}")
            sys.stdout.flush()
    try:
        rlist, _, _ = select.select([sys.stdin.fileno(), wake_r], [], [], 0.2)
    except InterruptedError:
        continue
    if wake_r in rlist:
        try:
            os.read(wake_r, 1024)
        except OSError:
            pass
    if sys.stdin.fileno() in rlist:
        cmd = os.read(sys.stdin.fileno(), 64)
        if b"x" in cmd and in_alt:
            in_alt = False
            r, c = _winsize()
            sys.stdout.write(f"\\x1b[?1049lPOST_ALT_NORMAL_{r}x{c}\\r\\n")
            sys.stdout.flush()
        if not cmd or b"q" in cmd:
            break
""",
        encoding="utf-8",
    )
    fake_alt_then_inline.chmod(fake_alt_then_inline.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 24, 80)
    vt = _MiniVT100(rows=24, cols=80)
    vt.feed(b"SHELL_HISTORY_TOP\r\n")

    env = {
        **os.environ,
        "JETSKI_REAL_BIN": str(fake_alt_then_inline),
        "JETSKI_DATA_DIR": str(tmp_path / "jetski_alt_exit_data"),
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

    def _drain(duration: float) -> None:
        end_at = time.monotonic() + duration
        while True:
            rem = end_at - time.monotonic()
            if rem <= 0:
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.02, rem))
            if ready:
                try:
                    chunk = os.read(master_fd, 16384)
                    if chunk:
                        vt.feed(chunk)
                except OSError:
                    break

    try:
        # 1. Wait for alt-screen at 24x80 (`ALT_VIEW_22x80`)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if "ALT_VIEW_22x80" in lines[0] and "Tokens" in lines[22]:
                break
        else:
            raise AssertionError("Timed out waiting for ALT_VIEW_22x80")

        # 2. Resize terminal while in alt-screen to 44x120
        vt.resize(44, 120)
        set_pty_winsize(master_fd, 44, 120)
        os.kill(proc.pid, signal.SIGWINCH)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if len(lines) == 44 and "ALT_VIEW_42x120" in lines[0] and "Tokens" in lines[42]:
                break
        else:
            raise AssertionError("Timed out waiting for ALT_VIEW_42x120")

        # 3. Tell child to exit `?1049l` back to normal screen while staying alive
        os.write(master_fd, b"x\n")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if (
                any("POST_ALT_NORMAL_42x120" in l for l in lines[:42])
                and "Tokens" in lines[42]
                and "Explore" in lines[43]
            ):
                break
        else:
            raise AssertionError(
                "Timed out waiting for POST_ALT_NORMAL_42x120:\n"
                + "\n".join(f"{i + 1:02d}: {l}" for i, l in enumerate(vt.snapshot_lines()))
            )

        # Verify ZERO zombie startup bar at old rows 23..24 (0-indexed 22..23) or anywhere in 0..41
        lines = vt.snapshot_lines()
        for r in range(42):
            assert "Tokens" not in lines[r], f"Zombie startup bar on row {r + 1}: {lines[r]!r}"
            assert "Explore" not in lines[r], f"Zombie category line on row {r + 1}: {lines[r]!r}"
        assert any("SHELL_HISTORY_TOP" in l for l in lines[:10])
        assert any("PRE_ALT_BANNER_LINE" in l for l in lines[:10])
        assert "Tokens" in lines[42]
        assert "Explore" in lines[43]

        os.write(master_fd, b"q\n")
        exit_deadline = time.monotonic() + 5.0
        while proc.poll() is None and time.monotonic() < exit_deadline:
            _drain(0.05)
        assert proc.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_normal_buffer_shrink_reflow_tiny_collapse_and_inband_stdin_resize(
    tmp_path: Path,
) -> None:
    """
    Verify remaining normal-buffer & protocol resize edge cases:
    1. Normal-screen vertical shrink (`preserve_normal_cursor_above_footer=True`) scrolls
       bottom-clamped prompt lines safely out of the new footer band without disturbing
       cursors already above `new_child_rows`.
    2. Normal-screen horizontal width shrink (`prev_normal_footer_vis_lens`) expands clear
       range to erase reflow-wrapped footer lines.
    3. End-to-end PTY run shrinks below `MIN_TERMINAL_ROWS_FOR_FOOTER` (`30x100 -> 4x70`),
       recovers to `26x95`, and rewrites DEC Mode 2048 (`\\x1b[48;...t`) and `XTWINOPS`
       (`\\x1b[8;...t`) events arriving on `stdin` to `child_rows` (`24`).
    """
    # 1a. Bottom-clamped cursor in normal screen during shrink (30 -> 20, new_child_rows=18):
    vt_bottom = _MiniVT100(rows=30, cols=80)
    vt_bottom.feed(b"\x1b[28;1HPROMPT_AT_BOTTOM")
    vt_bottom.resize(20, 80)  # Terminal clamps cursor & bottom line or leaves line at row 20
    vt_bottom.grid[19][:16] = list("CLAMPED_PROMPT_X")
    vt_bottom.row = 19
    seq_shrink = build_resize_transition_sequence(
        prev_rows=30,
        prev_use_footer=True,
        new_rows=20,
        new_use_footer=True,
        preserve_normal_cursor_above_footer=True,
    )
    vt_bottom.feed(seq_shrink)
    snap_b = vt_bottom.snapshot_lines()
    assert snap_b[17] == "CLAMPED_PROMPT_X"  # Scrolled safely up to row 18 (0-idx 17)!
    assert snap_b[18] == "" and snap_b[19] == ""
    assert vt_bottom.row == 17

    # 1b. Cursor already above footer (row 5) during normal-screen shrink: zero unwanted scroll!
    vt_top = _MiniVT100(rows=30, cols=80)
    vt_top.feed(b"\x1b[1;1HTOP_LINE_1\x1b[5;1HPROMPT_ROW_5")
    vt_top.resize(20, 80)
    vt_top.feed(seq_shrink)
    snap_t = vt_top.snapshot_lines()
    assert snap_t[0] == "TOP_LINE_1"
    assert snap_t[4] == "PROMPT_ROW_5"
    assert vt_top.row == 4

    # 2. Width-only narrow in normal screen (48x140 -> 48x60) clears wrapped footer rows (45..48)
    seq_narrow = build_resize_transition_sequence(
        prev_rows=48,
        prev_use_footer=True,
        new_rows=48,
        new_use_footer=True,
        preserve_normal_cursor_above_footer=True,
        new_cols=60,
        prev_normal_footer_vis_lens=(115, 95),
    )
    assert seq_narrow == b"\x1b7\x1b[0m\x1b[1;46r\x1b[45;1H\x1b[J\x1b8"

    # 3. End-to-end PTY test: 30x100 -> tiny 4x70 (with ?1049h/?1049l) -> 26x95 + in-band stdin resize
    fake_tiny_and_inband = tmp_path / "fake_tiny_and_inband"
    fake_tiny_and_inband.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, select, signal, struct, sys, termios, tty

data_dir = os.environ.get("AGY_DATA_DIR")
if data_dir:
    logs = pathlib.Path(data_dir) / "brain" / "tiny-inband-001" / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    step = {
        "step_index": 1,
        "source": "MODEL",
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "input_tokens": 1800,
        "cache_read_tokens": 3200,
        "output_tokens": 250,
        "tool_calls": [{"name": "edit_file", "args": {"Path": "src/main.py"}}],
    }
    (logs / "transcript.jsonl").write_text(json.dumps(step) + "\\n", encoding="utf-8")

tty.setraw(sys.stdin.fileno())
wake_r, wake_w = os.pipe()
os.set_blocking(wake_r, False)
os.set_blocking(wake_w, False)
resized = [True]

def _on_winch(_s, _f):
    resized[0] = True
    try:
        os.write(wake_w, b"W")
    except OSError:
        pass

signal.signal(signal.SIGWINCH, _on_winch)

def _winsize():
    packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
    r, c, _, _ = struct.unpack("HHHH", packed)
    return max(1, r), max(1, c)

while True:
    if resized[0]:
        resized[0] = False
        r, c = _winsize()
        if r == 4:
            # Exercise alt-screen enter/exit while footer is disabled (< 6 rows)
            sys.stdout.write(f"\\x1b[?1049hTINY_ALT\\x1b[?1049l\\r\\nTINY_NORM_{r}x{c}\\r\\n")
        else:
            sys.stdout.write(f"\\r\\nSIZE_NOW_{r}x{c}\\r\\n")
        sys.stdout.flush()
    try:
        rlist, _, _ = select.select([sys.stdin.fileno(), wake_r], [], [], 0.2)
    except InterruptedError:
        continue
    if wake_r in rlist:
        try:
            os.read(wake_r, 1024)
        except OSError:
            pass
    if sys.stdin.fileno() in rlist:
        data = os.read(sys.stdin.fileno(), 256)
        if not data or b"q" in data:
            break
        escaped = data.decode("latin1").replace("\\x1b", "ESC")
        sys.stdout.write(f"\\r\\nSTDIN_SEEN<{escaped}>\\r\\n")
        sys.stdout.flush()
""",
        encoding="utf-8",
    )
    fake_tiny_and_inband.chmod(fake_tiny_and_inband.stat().st_mode | stat.S_IXUSR)

    master_fd, slave_fd = pty.openpty()
    set_pty_winsize(slave_fd, 30, 100)
    vt = _MiniVT100(rows=30, cols=100)

    env = {
        **os.environ,
        "AGY_REAL_BIN": str(fake_tiny_and_inband),
        "AGY_DATA_DIR": str(tmp_path / "agy_tiny_inband_data"),
        "TERM": "xterm-256color",
    }
    proc = subprocess.Popen(
        [str(BIN_AGY), "--visualize-tokens", "--viz-heuristic-only"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    def _drain(duration: float) -> None:
        end_at = time.monotonic() + duration
        while True:
            rem = end_at - time.monotonic()
            if rem <= 0:
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.02, rem))
            if ready:
                try:
                    chunk = os.read(master_fd, 16384)
                    if chunk:
                        vt.feed(chunk)
                except OSError:
                    break

    try:
        # Phase 1: Initial 30x100 (`child_rows=28`)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if any("SIZE_NOW_28x100" in l for l in lines[:28]) and "Tokens" in lines[28]:
                break
        else:
            raise AssertionError("Timed out waiting for SIZE_NOW_28x100")

        # Phase 2: Shrink to 4x70 (< MIN_TERMINAL_ROWS_FOR_FOOTER -> child gets full 4x70, no footer)
        vt.resize(4, 70)
        set_pty_winsize(master_fd, 4, 70)
        os.kill(proc.pid, signal.SIGWINCH)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if any("TINY_NORM_4x70" in l for l in lines):
                break
        else:
            raise AssertionError("Timed out waiting for TINY_NORM_4x70")

        lines_tiny = vt.snapshot_lines()
        assert all("Tokens" not in l for l in lines_tiny)

        # Phase 3: Expand back to 26x95 (`child_rows=24`, footer resumes on rows 25..26)
        vt.resize(26, 95)
        set_pty_winsize(master_fd, 26, 95)
        os.kill(proc.pid, signal.SIGWINCH)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if (
                len(lines) == 26
                and any("SIZE_NOW_24x95" in l for l in lines[:24])
                and "Tokens" in lines[24]
                and "Code" in lines[25]
            ):
                break
        else:
            raise AssertionError("Timed out waiting for SIZE_NOW_24x95 + recovered footer")

        # Phase 4: Send DEC Mode 2048 + XTWINOPS + CPR on stdin -> rewritten to child_rows=24
        os.write(master_fd, b"\x1b[48;26;95;520;950t\x1b[8;26;95t\x1b[26;10R")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _drain(0.03)
            lines = vt.snapshot_lines()
            if any(
                "STDIN_SEEN<ESC[48;24;95;480;950tESC[8;24;95tESC[24;10R>" in l
                for l in lines[:24]
            ):
                break
        else:
            raise AssertionError(
                "Timed out waiting for rewritten stdin events:\n"
                + "\n".join(f"{i + 1:02d}: {l}" for i, l in enumerate(vt.snapshot_lines()))
            )

        os.write(master_fd, b"q")
        exit_deadline = time.monotonic() + 5.0
        while proc.poll() is None and time.monotonic() < exit_deadline:
            _drain(0.05)
        assert proc.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)






