"""PTY terminal wrapper reserving a 2-line bottom status bar while running `agy` or `jetski` cleanly."""

from __future__ import annotations

import errno
import fcntl
import math
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import tty
from typing import Callable, Sequence

from agy_token_viz.renderer import (
    render_single_status_line,
    render_status_lines,
    should_use_color,
    visible_len,
)
from agy_token_viz.state import SessionTokenState


BAR_ROWS = 2
MIN_TERMINAL_ROWS_FOR_FOOTER = 6
SYNC_UPDATE_TIMEOUT_SEC = 0.25
DEFAULT_RESIZE_SETTLE_SEC = 0.08

# Detect child sequences that clear the screen, switch screen buffers, or reset terminal
# state (requiring the 2-line footer at `rows - 1..rows` to be repainted).
_CLEAR_OR_BUFFER_RE = re.compile(
    rb"\x1b\[(?:\?(?:[0-9]+;)*(?:1049|1047|47)(?:;[0-9]+)*[hl]|[012]?J|!p)|\x1bc"
)

# Track alternate screen buffer entry ('h') and exit ('l')
_ALT_BUFFER_RE = re.compile(
    rb"\x1b\[\?(?:[0-9]+;)*(?:1049|1047|47)(?:;[0-9]+)*([hl])"
)

# Track DEC Mode 2026 synchronized output begin ('h') and end ('l')
_SYNC_UPDATE_RE = re.compile(rb"\x1b\[\?(?:[0-9]+;)*2026(?:;[0-9]+)*([hl])")

# Child sequences that switch/reset screen buffers (`?1049h/l`, `DECSTR`, `RIS`) or
# configure `DECSTBM` top/bottom scroll margins (`\x1b[Pt;Pbr`).
_SCROLL_GUARD_RE = re.compile(
    rb"(?P<reset>\x1b\[(?:\?(?:[0-9]+;)*(?:1049|1047|47)(?:;[0-9]+)*[hl]|!p)|\x1bc)"
    rb"|\x1b\[(?P<decstbm>[0-9;]*)r"
)

# SGR mouse events (`\x1b[<btn;col;rowM` or `\x1b[<btn;col;rowm`)
_SGR_MOUSE_RE = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

# Terminal-to-application reports carrying row/height coordinates on stdin:
# - XTWINOPS text-area size report: `\x1b[8;rows;colst`
# - DEC Mode 2048 in-band resize notification: `\x1b[48;rows;cols;height_px;width_pxt`
# - ANSI DSR/CPR cursor position report: `\x1b[row;colR`
_TERMINAL_SIZE_OR_CPR_INPUT_RE = re.compile(
    rb"\x1b\[(?:"
    rb"8;(?P<xt_r>\d+);(?P<xt_c>\d+)t"
    rb"|48;(?P<ib_r>\d+);(?P<ib_c>\d+)(?P<ib_rest>(?:;\d+)*)t"
    rb"|(?P<cpr_r>\d+);(?P<cpr_c>\d+)R"
    rb")"
)

# Incomplete trailing CSI input prefix (`\x1b[<...` or `\x1b[<digits>;...`, >= 3 bytes)
_INCOMPLETE_STDIN_CSI_RE = re.compile(rb"\x1b\[[<0-9;]+$")

# Child absolute vertical cursor positioning (`CUP` `\x1b[row;colH`, `HVP` `\x1b[row;colf`,
# `VPA` `\x1b[rowd`) that could otherwise land on reserved footer rows (`row > child_rows`)
# during an in-flight pre-shrink frame or a `\x1b[999;999H\x1b[6n` size probe.
_CHILD_ABS_ROW_RE = re.compile(
    rb"\x1b\[(?:(?P<cup_r>\d+)(?P<cup_c>;\d*)?(?P<cup_f>[Hf])|(?P<vpa_r>\d+)d)"
)

# Screen buffer mode transitions (`RIS` `\x1bc` or `?1049h/l`, `?1047h/l`, `?47h/l`)
_MODE_TRANSITION_RE = re.compile(
    rb"\x1bc|\x1b\[\?(?:[0-9]+;)*(?:1049|1047|47)(?:;[0-9]+)*([hl])"
)


def _parse_decstbm_params(params: bytes) -> tuple[int, int | None] | None:
    """
    Parse `DECSTBM` (`CSI Pt ; Pb r`) parameter bytes into `(top, bottom)`.

    Returns `(top, bottom)` where `top >= 1` and `bottom` is `None` when omitted or `0`
    (meaning VT100/xterm default full-terminal bottom margin), or returns `None` if
    `params` has more than two semicolon-separated fields.
    """
    if not params:
        return 1, None
    parts = params.split(b";")
    if len(parts) > 2:
        return None
    top_raw = int(parts[0]) if parts[0] else 0
    top = 1 if top_raw == 0 else top_raw
    if len(parts) == 1:
        return top, None
    bot_raw = int(parts[1]) if parts[1] else 0
    bottom = None if bot_raw == 0 else bot_raw
    return top, bottom


def inject_scroll_region_guards(
    data: bytes,
    child_rows: int,
    *,
    active_region: list[tuple[int, int] | None] | None = None,
) -> bytes:
    """
    Keep `DECSTBM` scroll margins confined within `1..child_rows` so child scrolling never
    reaches the reserved 2-line footer, while preserving valid sub-viewport scroll regions
    (e.g. `\\x1b[1;22r` used by Jetski/Ultraviolet to scroll generated text above the
    sticky user prompt area).

    - Buffer switches / resets (`\\x1b[?1049h/l`, `\\x1b[!p`, `\\x1bc`): append
      `\\x1b7\\x1b[1;{child_rows}r\\x1b8` so margins are confined without disturbing the
      restored/current cursor position.
    - Valid child sub-regions (`1 <= top < bottom <= child_rows`): pass through untouched.
    - Margin resets (`\\x1b[r`, `\\x1b[;r`, `0` bottom) or out-of-bounds bottom margins
      (`bottom > child_rows`): clamp inline to `\\x1b[{clamped_top};{child_rows}r`.
    """
    if not data or not _SCROLL_GUARD_RE.search(data):
        return data

    max_row = max(1, child_rows)
    guard = f"\x1b7\x1b[1;{max_row}r\x1b8".encode("ascii")
    region_state = active_region if active_region is not None else [None]

    def _replace(match: re.Match[bytes]) -> bytes:
        if match.group("reset") is not None:
            region_state[0] = None
            return match.group(0) + guard

        parsed = _parse_decstbm_params(match.group("decstbm"))
        if parsed is None:
            return match.group(0)

        top, bottom = parsed
        if bottom is not None and bottom <= max_row:
            if top < bottom:
                region_state[0] = None if (top == 1 and bottom == max_row) else (top, bottom)
            return match.group(0)

        clamped_bottom = max_row
        clamped_top = top if top < clamped_bottom else 1
        region_state[0] = None if clamped_top == 1 else (clamped_top, clamped_bottom)
        return f"\x1b[{clamped_top};{clamped_bottom}r".encode("ascii")

    return _SCROLL_GUARD_RE.sub(_replace, data)


def clamp_child_cursor_positions(data: bytes, child_rows: int) -> bytes:
    """
    Clamp explicit vertical cursor targets (`CUP` `H`, `HVP` `f`, `VPA` `d`) with
    `row > child_rows` to `child_rows` so:
    1. In-flight child frames rendered right before a window shrink never overwrite the
       reserved 2-line bottom footer (`rows - 1 .. rows`).
    2. Sub-tools probing terminal height via `\\x1b[999;999H\\x1b[6n` stop at `child_rows`
       and receive `\\x1b[{child_rows};{cols}R` instead of `outer_rows`.
    """
    if not data or b"\x1b[" not in data:
        return data

    max_row = max(1, child_rows)

    def _replace(match: re.Match[bytes]) -> bytes:
        cup_r = match.group("cup_r")
        if cup_r is not None:
            row = int(cup_r)
            if row <= max_row:
                return match.group(0)
            col_part = match.group("cup_c") or b""
            final = match.group("cup_f")
            return f"\x1b[{max_row}".encode("ascii") + col_part + final

        vpa_r = match.group("vpa_r")
        row = int(vpa_r)
        if row <= max_row:
            return match.group(0)
        return f"\x1b[{max_row}d".encode("ascii")

    return _CHILD_ABS_ROW_RE.sub(_replace, data)


def erase_normal_footer_before_alt_screen(
    data: bytes,
    *,
    in_alt_screen: list[bool],
    normal_footer_state: list[tuple[int, int, int] | None],
) -> bytes:
    """
    Track normal/alternate screen transitions and erase any active normal-buffer footer
    *immediately before* entering `?1049h` / `?1047h` / `?47h`.

    Without this, the terminal saves the normal screen buffer with the startup/previous
    token bar baked into `(old_rows - 1, old_rows)`; if the user resizes the window while
    in alt-screen and later exits (`?1049l`), that stale bar reappears stranded in the
    middle of the normal screen buffer.
    """
    if not data or b"\x1b" not in data or not _MODE_TRANSITION_RE.search(data):
        return data

    def _replace(match: re.Match[bytes]) -> bytes:
        raw = match.group(0)
        if raw == b"\x1bc":
            in_alt_screen[0] = False
            normal_footer_state[0] = None
            return raw

        entering_alt = match.group(1) == b"h"
        if entering_alt:
            was_alt = in_alt_screen[0]
            in_alt_screen[0] = True
            if not was_alt and normal_footer_state[0] is not None:
                nf_rows, _, _ = normal_footer_state[0]
                normal_footer_state[0] = None
                erase_seq = (
                    "\x1b7\x1b[0m"
                    f"\x1b[{max(1, nf_rows - 1)};1H\x1b[2K"
                    f"\x1b[{max(1, nf_rows)};1H\x1b[2K"
                    "\x1b[r\x1b8"
                ).encode("ascii")
                return erase_seq + raw
            return raw

        in_alt_screen[0] = False
        return raw

    return _MODE_TRANSITION_RE.sub(_replace, data)


def filter_footer_mouse_events(
    data: bytes,
    child_rows: int,
    *,
    suppressing_click: list[bool] | None = None,
) -> bytes:
    """
    Filter or clamp SGR mouse events (`\\x1b[<btn;col;rowM/m`) targeting the reserved
    2-line bottom footer (`row > child_rows`) so clicks on the status bar never trigger
    TUI controls while wheel scrolling and active drag-releases stay smooth.
    """
    if not data or b"\x1b[<" not in data:
        return data

    max_row = max(1, child_rows)
    state = suppressing_click if suppressing_click is not None else [False]

    def _replace(match: re.Match[bytes]) -> bytes:
        btn = int(match.group(1))
        col = int(match.group(2))
        row = int(match.group(3))
        final = match.group(4)

        is_wheel = (btn & 64) != 0
        is_motion = (btn & 32) != 0

        if row <= max_row:
            if final == b"M" and not is_wheel and not is_motion:
                state[0] = False
            elif final == b"m":
                state[0] = False
            return match.group(0)

        if is_wheel:
            return f"\x1b[<{btn};{col};{max_row}".encode("ascii") + final

        if is_motion:
            if state[0]:
                return b""
            return f"\x1b[<{btn};{col};{max_row}".encode("ascii") + final

        if final == b"M":
            state[0] = True
            return b""

        if state[0]:
            state[0] = False
            return b""
        return f"\x1b[<{btn};{col};{max_row}m".encode("ascii")

    return _SGR_MOUSE_RE.sub(_replace, data)


def rewrite_terminal_input_events(
    data: bytes,
    child_rows: int,
    *,
    suppressing_click: list[bool] | None = None,
) -> bytes:
    """
    Adjust terminal-to-child input sequences on `stdin` when the 2-line footer is active:
    - Filter/clamp SGR mouse events (`\\x1b[<btn;col;rowM/m`) over footer rows.
    - Rewrite `XTWINOPS` (`\\x1b[8;rows;colst`) and DEC Mode 2048 in-band resize events
      (`\\x1b[48;rows;cols;px_h;px_wt`) to report `child_rows` (and proportionally scaled
      pixel height) so TUIs never size themselves to `outer_rows` over the footer.
    - Clamp `CPR` (`\\x1b[row;colR`) replies with `row > child_rows` to `child_rows`
      (leaving `1 <= row <= child_rows`, including modified F3 keys `\\x1b[1;<mod>R`, intact).
    """
    if not data or b"\x1b[" not in data:
        return data

    out = filter_footer_mouse_events(
        data, child_rows, suppressing_click=suppressing_click
    )
    if not out or b"\x1b[" not in out:
        return out

    max_row = max(1, child_rows)

    def _replace(match: re.Match[bytes]) -> bytes:
        xt_r = match.group("xt_r")
        if xt_r is not None:
            orig_r = int(xt_r)
            new_r = max_row if orig_r > 0 else 0
            return f"\x1b[8;{new_r};".encode("ascii") + match.group("xt_c") + b"t"

        ib_r = match.group("ib_r")
        if ib_r is not None:
            orig_r = int(ib_r)
            new_r = max_row if orig_r > 0 else 0
            rest = match.group("ib_rest") or b""
            if rest and orig_r > 0:
                fields = [part for part in rest.split(b";") if part]
                if fields:
                    px_h = int(fields[0])
                    scaled_px_h = (
                        max(1, int(round(px_h * (max_row / orig_r))))
                        if px_h > 0
                        else 0
                    )
                    fields[0] = str(scaled_px_h).encode("ascii")
                    rest = b";" + b";".join(fields)
            return (
                f"\x1b[48;{new_r};".encode("ascii")
                + match.group("ib_c")
                + rest
                + b"t"
            )

        cpr_r = int(match.group("cpr_r"))
        if cpr_r <= max_row:
            return match.group(0)
        return f"\x1b[{max_row};".encode("ascii") + match.group("cpr_c") + b"R"

    return _TERMINAL_SIZE_OR_CPR_INPUT_RE.sub(_replace, out)


def get_terminal_size(fd: int) -> tuple[int, int]:
    """Query `(rows, cols)` of a terminal file descriptor, defaulting to `(24, 80)` if unset."""
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        rows, cols, _, _ = struct.unpack("HHHH", packed)
        if rows > 0 and cols > 0:
            return rows, cols
    except OSError:
        pass
    import shutil

    fallback = shutil.get_terminal_size(fallback=(80, 24))
    return max(24, fallback.lines) if fallback.lines <= 0 else fallback.lines, (
        max(80, fallback.columns) if fallback.columns <= 0 else fallback.columns
    )


def set_pty_winsize(
    fd: int,
    rows: int,
    cols: int,
    *,
    child_pid: int | None = None,
) -> None:
    """
    Set `(rows, cols)` on a PTY descriptor and ensure `SIGWINCH` reaches the child.

    When `slave_fd` is configured as the child session's controlling terminal via
    `TIOCSCTTY`, `TIOCSWINSZ` automatically delivers `SIGWINCH` to the PTY's foreground
    process group (`TIOCGPGRP > 0`). If no foreground process group is attached (e.g.
    restricted environments), fall back to signaling `child_pid`'s process group directly.
    """
    try:
        packed = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass

    if child_pid is not None and child_pid > 0:
        has_fg_pgrp = False
        if hasattr(termios, "TIOCGPGRP"):
            try:
                pgrp_bytes = fcntl.ioctl(fd, termios.TIOCGPGRP, struct.pack("i", 0))
                has_fg_pgrp = struct.unpack("i", pgrp_bytes)[0] > 0
            except OSError:
                has_fg_pgrp = False
        if not has_fg_pgrp:
            try:
                os.killpg(child_pid, signal.SIGWINCH)
            except OSError:
                try:
                    os.kill(child_pid, signal.SIGWINCH)
                except OSError:
                    pass


def _attach_controlling_tty() -> None:
    """Attach fd 0 (`slave_fd`) as the child session's controlling terminal after `setsid()`."""
    if hasattr(termios, "TIOCSCTTY"):
        try:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        except OSError:
            pass


def build_resize_transition_sequence(
    prev_rows: int,
    prev_use_footer: bool,
    new_rows: int,
    new_use_footer: bool,
    *,
    preserve_normal_cursor_above_footer: bool = False,
    new_cols: int | None = None,
    prev_normal_footer_vis_lens: tuple[int, int] | None = None,
) -> bytes:
    """
    Build the immediate escape sequence applied on `SIGWINCH` before forwarding the new
    `(child_rows, cols)` window size to the child PTY:

    1. In normal-screen mode (`preserve_normal_cursor_above_footer=True`), if the child
       viewport shrank vertically (`new_child_rows < prev_child_rows`) so the terminal may
       have clamped the child's cursor/prompt into `new_rows - 1 .. new_rows`, execute
       `\\x1b7\\x1b[r\\x1b8\\x1bD\\x1bD\\x1b[2A` to scroll any bottom-clamped lines safely
       up into `1..new_child_rows` without disturbing cursors already at `<= new_child_rows`.
    2. Re-confine `DECSTBM` scroll margins (`\\x1b[1;{new_child_rows}r`, or `\\x1b[r` when
       footer is disabled) because terminal emulators reset `DECSTBM` to `1..new_rows` on
       window resize.
    3. Erase previously painted footer rows (`prev_child_rows + 1 .. prev_rows`), any
       normal-screen footer lines that reflow-wrapped on width shrink, newly exposed
       viewport rows (`prev_child_rows + 1 .. new_child_rows`), and the new reserved footer
       rows (`new_child_rows + 1 .. new_rows`) with reset SGR (`\\x1b[0m`).
    """
    if not prev_use_footer and not new_use_footer:
        return b""

    prev_child_rows = (
        max(1, prev_rows - BAR_ROWS) if prev_use_footer else max(1, prev_rows)
    )
    new_child_rows = (
        max(1, new_rows - BAR_ROWS) if new_use_footer else max(1, new_rows)
    )

    parts: list[str] = []
    if (
        preserve_normal_cursor_above_footer
        and new_use_footer
        and new_child_rows < prev_child_rows
    ):
        parts.append(f"\x1b7\x1b[r\x1b8" + ("\x1bD" * BAR_ROWS) + f"\x1b[{BAR_ROWS}A")

    parts.append("\x1b7\x1b[0m")
    if new_use_footer:
        parts.append(f"\x1b[1;{new_child_rows}r")
        clear_start = min(prev_child_rows + 1, new_child_rows + 1)
        if (
            preserve_normal_cursor_above_footer
            and new_rows >= prev_rows
            and new_cols is not None
            and new_cols > 0
            and prev_normal_footer_vis_lens is not None
        ):
            vis1, vis2 = prev_normal_footer_vis_lens
            wrapped_rows = (
                (math.ceil(vis1 / new_cols) if vis1 > 0 else 1)
                + (math.ceil(vis2 / new_cols) if vis2 > 0 else 1)
            )
            if wrapped_rows > BAR_ROWS:
                reflow_clear_start = new_rows - wrapped_rows + 1
                clear_start = min(clear_start, reflow_clear_start)
        clear_start = max(1, min(new_rows, clear_start))
        parts.append(f"\x1b[{clear_start};1H\x1b[J")
    else:
        parts.append("\x1b[r")
        if prev_child_rows + 1 <= new_rows:
            clear_start = max(1, prev_child_rows + 1)
            parts.append(f"\x1b[{clear_start};1H\x1b[J")
    parts.append("\x1b8")
    return "".join(parts).encode("ascii")


def split_complete_terminal_bytes(data: bytes) -> tuple[bytes, bytes]:
    """
    Split `data` into `(complete_prefix, trailing_incomplete_suffix)` so we never inject
    our status-bar escape codes in the middle of a child's multi-byte UTF-8 char or ANSI CSI/OSC.
    """
    if not data:
        return b"", b""

    # 1. Check for incomplete trailing ANSI escape (within last 64 bytes)
    tail_window = min(len(data), 64)
    last_esc = data.rfind(b"\x1b", len(data) - tail_window)
    if last_esc != -1:
        seq = data[last_esc:]
        if len(seq) == 1:
            return data[:last_esc], seq
        second = seq[1:2]
        if second == b"[":
            # CSI: \x1b [ <params 0x30-0x3f> <intermediates 0x20-0x2f> <final 0x40-0x7e>
            done = False
            for b in seq[2:]:
                if 0x40 <= b <= 0x7E:
                    done = True
                    break
            if not done:
                return data[:last_esc], seq
        elif second == b"]":
            # OSC: terminated by BEL (\x07) or ST (\x1b\\)
            if b"\x07" not in seq and b"\x1b\\" not in seq:
                return data[:last_esc], seq
        elif second in (b"(", b")", b"*", b"+", b"#", b" "):
            if len(seq) < 3:
                return data[:last_esc], seq

    # 2. Check for incomplete trailing UTF-8 sequence (1..3 trailing continuation bytes)
    for back in range(1, min(4, len(data)) + 1):
        b = data[-back]
        if (b & 0b1000_0000) == 0:
            break  # ASCII
        if (b & 0b1100_0000) == 0b1100_0000:
            # Leading UTF-8 byte
            if (b & 0b1110_0000) == 0b1100_0000:
                expected = 2
            elif (b & 0b1111_0000) == 0b1110_0000:
                expected = 3
            else:
                expected = 4
            if back < expected:
                split_at = len(data) - back
                return data[:split_at], data[split_at:]
            break

    return data, b""


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    total = len(payload)
    sent = 0
    while sent < total:
        try:
            n = os.write(fd, view[sent:])
            sent += n
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EINTR):
                continue
            break


class PTYVisualizerRunner:
    """
    Executes `agy` inside a managed PTY with `(rows - 2, cols)` while rendering
    the live 2-line tqdm-style token breakdown in the bottom 2 rows.
    """

    def __init__(
        self,
        cmd: Sequence[str],
        state: SessionTokenState,
        use_color: bool | None = None,
        min_refresh_interval: float = 0.05,
        resize_settle_delay: float = DEFAULT_RESIZE_SETTLE_SEC,
    ) -> None:
        self._cmd = list(cmd)
        self._state = state
        self._use_color = should_use_color() if use_color is None else use_color
        self._min_refresh_interval = min_refresh_interval
        self._resize_settle_delay = max(0.0, resize_settle_delay)
        self._last_footer_vis_lens: tuple[int, int] = (0, 0)

    def run(
        self,
        on_before_spawn: Callable[[], None] | None = None,
        on_process_spawn: Callable[[int], None] | None = None,
    ) -> int:
        """Spawn `agy` and manage terminal output until exit; returns child exit code."""
        stdin_tty = os.isatty(sys.stdin.fileno())
        stdout_tty = os.isatty(sys.stdout.fileno())
        stderr_tty = os.isatty(sys.stderr.fileno())

        if not (stdin_tty and stdout_tty):
            return self._run_non_interactive(
                stderr_tty=stderr_tty,
                on_before_spawn=on_before_spawn,
                on_process_spawn=on_process_spawn,
            )
        return self._run_pty_interactive(
            on_before_spawn=on_before_spawn,
            on_process_spawn=on_process_spawn,
        )

    def _run_non_interactive(
        self,
        stderr_tty: bool,
        on_before_spawn: Callable[[], None] | None = None,
        on_process_spawn: Callable[[int], None] | None = None,
    ) -> int:
        """Pass stdout straight through when piped/redirected; optionally show stderr tqdm line."""
        wakeup_r, wakeup_w = os.pipe()
        os.set_blocking(wakeup_r, False)
        os.set_blocking(wakeup_w, False)

        def _notify() -> None:
            try:
                os.write(wakeup_w, b"\x01")
            except OSError:
                pass

        stderr_fd = os.dup(sys.stderr.fileno())
        os.set_inheritable(stderr_fd, False)

        prev_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _on_sigwinch(_signum: int, _frame: object) -> None:
            _notify()

        self._state.set_on_change(_notify)
        if stderr_tty:
            signal.signal(signal.SIGWINCH, _on_sigwinch)
        if on_before_spawn is not None:
            on_before_spawn()

        proc = subprocess.Popen(self._cmd)
        if on_process_spawn is not None:
            on_process_spawn(proc.pid)
        drew_stderr_bar = False
        prev_stderr_vis_len = 0
        prev_stderr_cols = 0

        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                rlist, _, _ = select.select([wakeup_r], [], [], 0.2)
                if rlist:
                    try:
                        os.read(wakeup_r, 1024)
                    except OSError:
                        pass
                if stderr_tty:
                    _, cols = get_terminal_size(stderr_fd)
                    render_cols = max(1, cols - 1) if cols > 1 else 1
                    snap = self._state.snapshot()
                    line = render_single_status_line(
                        snap, cols=render_cols, use_color=self._use_color
                    )
                    erase_prefix = "\r\x1b[2K"
                    if (
                        drew_stderr_bar
                        and prev_stderr_cols > 0
                        and cols < prev_stderr_cols
                        and prev_stderr_vis_len > cols
                    ):
                        extra_wrapped = (prev_stderr_vis_len - 1) // max(1, cols)
                        erase_prefix += "\x1b[1A\r\x1b[2K" * extra_wrapped
                    _write_all(stderr_fd, f"{erase_prefix}{line}".encode("utf-8"))
                    prev_stderr_vis_len = visible_len(line)
                    prev_stderr_cols = cols
                    drew_stderr_bar = True
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            rc = proc.wait()
        finally:
            self._state.set_on_change(None)
            if stderr_tty:
                signal.signal(signal.SIGWINCH, prev_sigwinch)
            if drew_stderr_bar:
                _, cols = get_terminal_size(stderr_fd)
                erase_seq = "\r\x1b[2K"
                if prev_stderr_vis_len > max(1, cols):
                    extra_wrapped = (prev_stderr_vis_len - 1) // max(1, cols)
                    erase_seq += "\x1b[1A\r\x1b[2K" * extra_wrapped
                _write_all(stderr_fd, erase_seq.encode("ascii"))
            os.close(stderr_fd)
            os.close(wakeup_r)
            os.close(wakeup_w)

        return int(proc.returncode or 0)

    def _build_footer_sequence(
        self,
        rows: int,
        cols: int,
        reapply_scroll_region: bool = False,
        scroll_region: tuple[int, int] | None = None,
    ) -> bytes:
        child_rows = max(1, rows - BAR_ROWS)
        # Leave 1 column of margin on the right edge when cols > 1 so writing row `rows`
        # never touches cell `(rows, cols)`, avoiding pending-wrap or non-xenl auto-scroll.
        render_cols = max(1, cols - 1) if cols > 1 else 1
        snap = self._state.snapshot()
        line1, line2 = render_status_lines(
            snap, cols=render_cols, use_color=self._use_color
        )
        self._last_footer_vis_lens = (visible_len(line1), visible_len(line2))

        # IMPORTANT: Save child cursor & SGR state (\x1b7) BEFORE resetting SGR (\x1b[0m)
        # and DECSTBM. Resetting SGR before `\x1b[2K` prevents active 24-bit RGB background
        # colors in alt-screen TUIs from bleeding via Background Color Erase (BCE); `\x1b8`
        # restores the child's saved cursor position and SGR attributes.
        parts: list[str] = ["\x1b7\x1b[0m"]
        if reapply_scroll_region:
            if scroll_region is not None:
                reg_bottom = min(scroll_region[1], child_rows)
                reg_top = scroll_region[0] if scroll_region[0] < reg_bottom else 1
            else:
                reg_top, reg_bottom = 1, child_rows
            parts.append(f"\x1b[{reg_top};{reg_bottom}r")
        parts.append(f"\x1b[{rows - 1};1H\x1b[2K{line1}")
        parts.append(f"\x1b[{rows};1H\x1b[2K{line2}")
        parts.append("\x1b8")
        return "".join(parts).encode("utf-8")

    def _run_pty_interactive(
        self,
        on_before_spawn: Callable[[], None] | None = None,
        on_process_spawn: Callable[[int], None] | None = None,
    ) -> int:
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        # Keep a dedicated non-inheritable duplicate of the controlling terminal fd so
        # any library that temporarily redirects fd 1/2 (e.g. C/FFI loaders) cannot
        # swallow PTY output on the main thread.
        tty_out_fd = os.dup(stdout_fd)
        os.set_inheritable(tty_out_fd, False)

        rows, cols = get_terminal_size(tty_out_fd)
        use_footer = rows >= MIN_TERMINAL_ROWS_FOR_FOOTER
        child_rows = max(1, rows - BAR_ROWS) if use_footer else rows

        master_fd, slave_fd = pty.openpty()
        set_pty_winsize(slave_fd, child_rows, cols)

        wakeup_r, wakeup_w = os.pipe()
        os.set_blocking(wakeup_r, False)
        os.set_blocking(wakeup_w, False)

        resize_flag = [False]
        state_dirty = [True]
        in_alt_screen = [False]
        normal_footer_state: list[tuple[int, int, int] | None] = [None]

        def _wake(code: bytes) -> None:
            try:
                os.write(wakeup_w, code)
            except OSError:
                pass

        def _on_state_change() -> None:
            state_dirty[0] = True
            _wake(b"S")

        prev_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _on_sigwinch(_signum: int, _frame: object) -> None:
            resize_flag[0] = True
            _wake(b"W")

        self._state.set_on_change(_on_state_change)
        signal.signal(signal.SIGWINCH, _on_sigwinch)

        if on_before_spawn is not None:
            on_before_spawn()

        orig_termios = termios.tcgetattr(stdin_fd)

        # Reserve 2 clean bottom lines in normal buffer so initial prompt sits above footer
        if use_footer:
            init_seq = (
                "\r\n\r\n\x1b[2A"  # Ensure 2 scrollable rows exist at bottom without losing history
                "\x1b7"
                f"\x1b[1;{child_rows}r"
                "\x1b8"
            ).encode("utf-8")
            _write_all(tty_out_fd, init_seq)
            _write_all(
                tty_out_fd,
                self._build_footer_sequence(rows, cols, reapply_scroll_region=False),
            )
            normal_footer_state[0] = (
                rows,
                self._last_footer_vis_lens[0],
                self._last_footer_vis_lens[1],
            )

        proc = subprocess.Popen(
            self._cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
            preexec_fn=_attach_controlling_tty,
        )
        os.close(slave_fd)
        if on_process_spawn is not None:
            on_process_spawn(proc.pid)

        pending_child_bytes = b""
        pending_stdin_bytes = b""
        last_bar_paint = time.monotonic()
        footer_drawn = use_footer
        resize_settle_deadline = 0.0
        in_sync_update = False
        sync_started_at = 0.0
        pending_region_restore = False
        active_scroll_region: list[tuple[int, int] | None] = [None]
        suppressing_footer_click = [False]

        def _track_sync_mode(raw: bytes, now_ts: float) -> None:
            nonlocal in_sync_update, sync_started_at
            reset_idx = raw.rfind(b"\x1bc")
            if reset_idx != -1:
                in_sync_update = False
                raw = raw[reset_idx + 2 :]
            for sync_m in _SYNC_UPDATE_RE.finditer(raw):
                if sync_m.group(1) == b"h":
                    in_sync_update = True
                    sync_started_at = now_ts
                else:
                    in_sync_update = False

        try:
            tty.setraw(stdin_fd)

            while True:
                # Detect silent / startup terminal size changes even if SIGWINCH coalesced
                polled_rows, polled_cols = get_terminal_size(tty_out_fd)
                if (polled_rows, polled_cols) != (rows, cols):
                    resize_flag[0] = True

                now_pre = time.monotonic()
                if resize_flag[0]:
                    timeout = 0.0
                elif resize_settle_deadline > 0.0 and now_pre < resize_settle_deadline:
                    timeout = min(0.05, max(0.005, resize_settle_deadline - now_pre))
                elif (
                    state_dirty[0]
                    or pending_region_restore
                    or in_sync_update
                    or (use_footer and not footer_drawn)
                ):
                    timeout = 0.05
                else:
                    timeout = 0.25
                try:
                    rlist, _, _ = select.select(
                        [master_fd, stdin_fd, wakeup_r], [], [], timeout
                    )
                except InterruptedError:
                    rlist = []

                if in_sync_update and (
                    time.monotonic() - sync_started_at
                ) >= SYNC_UPDATE_TIMEOUT_SEC:
                    in_sync_update = False

                if wakeup_r in rlist:
                    try:
                        os.read(wakeup_r, 1024)
                    except OSError:
                        pass

                cur_rows_top, cur_cols_top = get_terminal_size(tty_out_fd)
                if resize_flag[0] or (cur_rows_top, cur_cols_top) != (rows, cols):
                    resize_flag[0] = False
                    prev_rows = rows
                    prev_use_footer = use_footer
                    rows, cols = cur_rows_top, cur_cols_top
                    use_footer = rows >= MIN_TERMINAL_ROWS_FOR_FOOTER
                    child_rows = max(1, rows - BAR_ROWS) if use_footer else rows
                    active_scroll_region[0] = None

                    prev_vis_lens = (
                        (normal_footer_state[0][1], normal_footer_state[0][2])
                        if (not in_alt_screen[0] and normal_footer_state[0] is not None)
                        else None
                    )
                    transition_seq = build_resize_transition_sequence(
                        prev_rows=prev_rows,
                        prev_use_footer=prev_use_footer,
                        new_rows=rows,
                        new_use_footer=use_footer,
                        preserve_normal_cursor_above_footer=not in_alt_screen[0],
                        new_cols=cols,
                        prev_normal_footer_vis_lens=prev_vis_lens,
                    )
                    if transition_seq:
                        _write_all(tty_out_fd, transition_seq)
                    if not in_alt_screen[0]:
                        normal_footer_state[0] = None

                    set_pty_winsize(
                        master_fd, child_rows, cols, child_pid=proc.pid
                    )
                    footer_drawn = False
                    if use_footer:
                        pending_region_restore = True
                        resize_settle_deadline = (
                            time.monotonic() + self._resize_settle_delay
                        )
                    else:
                        pending_region_restore = False
                        resize_settle_deadline = 0.0

                if stdin_fd in rlist:
                    try:
                        user_bytes = os.read(stdin_fd, 4096)
                    except OSError:
                        user_bytes = b""
                    if user_bytes:
                        user_bytes = pending_stdin_bytes + user_bytes
                        pending_stdin_bytes = b""
                        while (
                            len(user_bytes) < 16384
                            and _INCOMPLETE_STDIN_CSI_RE.search(user_bytes) is not None
                        ):
                            try:
                                more_in, _, _ = select.select([stdin_fd], [], [], 0.002)
                                if not more_in:
                                    break
                                extra_in = os.read(stdin_fd, 4096)
                                if not extra_in:
                                    break
                                user_bytes += extra_in
                            except OSError:
                                break
                        if use_footer:
                            user_bytes = rewrite_terminal_input_events(
                                user_bytes,
                                child_rows,
                                suppressing_click=suppressing_footer_click,
                            )
                        if user_bytes:
                            _write_all(master_fd, user_bytes)

                if master_fd in rlist:
                    try:
                        chunk = os.read(master_fd, 16384)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            chunk = b""
                        else:
                            raise
                    if not chunk:
                        # Child closed PTY slave
                        if pending_child_bytes:
                            _track_sync_mode(pending_child_bytes, time.monotonic())
                            flushed = erase_normal_footer_before_alt_screen(
                                pending_child_bytes,
                                in_alt_screen=in_alt_screen,
                                normal_footer_state=normal_footer_state,
                            )
                            _write_all(tty_out_fd, flushed)
                            pending_child_bytes = b""
                        break

                    # Coalesce immediately available PTY bytes so single child frames
                    # larger than the 1024-byte kernel PTY buffer are processed together.
                    while len(chunk) < 65536:
                        try:
                            more, _, _ = select.select([master_fd], [], [], 0)
                            if not more:
                                break
                            extra = os.read(master_fd, 16384)
                            if not extra:
                                break
                            chunk += extra
                        except OSError:
                            break

                    combined = pending_child_bytes + chunk
                    complete, pending_child_bytes = split_complete_terminal_bytes(combined)
                    if complete:
                        now = time.monotonic()
                        was_alt = in_alt_screen[0]
                        _track_sync_mode(complete, now)
                        if use_footer:
                            guarded = clamp_child_cursor_positions(complete, child_rows)
                            guarded = inject_scroll_region_guards(
                                guarded,
                                child_rows,
                                active_region=active_scroll_region,
                            )
                            had_mode_transition = bool(
                                _MODE_TRANSITION_RE.search(complete)
                            )
                            guarded = erase_normal_footer_before_alt_screen(
                                guarded,
                                in_alt_screen=in_alt_screen,
                                normal_footer_state=normal_footer_state,
                            )
                            if had_mode_transition or in_alt_screen[0] != was_alt:
                                footer_drawn = False
                            chunk_needs_region = bool(_CLEAR_OR_BUFFER_RE.search(complete))
                            resize_settling = resize_flag[0] or (
                                resize_settle_deadline > 0.0
                                and now < resize_settle_deadline
                            )

                            if in_sync_update or resize_settling:
                                if chunk_needs_region:
                                    pending_region_restore = True
                                _write_all(tty_out_fd, guarded)
                            else:
                                if resize_settle_deadline > 0.0:
                                    resize_settle_deadline = 0.0
                                needs_region = chunk_needs_region or pending_region_restore
                                if (
                                    needs_region
                                    or not footer_drawn
                                    or (
                                        state_dirty[0]
                                        and (now - last_bar_paint)
                                        >= self._min_refresh_interval
                                    )
                                ):
                                    cur_rows, cur_cols = get_terminal_size(tty_out_fd)
                                    if (cur_rows, cur_cols) != (rows, cols):
                                        resize_flag[0] = True
                                        if chunk_needs_region:
                                            pending_region_restore = True
                                        _write_all(tty_out_fd, guarded)
                                    else:
                                        footer = self._build_footer_sequence(
                                            rows,
                                            cols,
                                            reapply_scroll_region=(
                                                needs_region or not footer_drawn
                                            ),
                                            scroll_region=active_scroll_region[0],
                                        )
                                        if guarded.endswith(b"\x1b[?2026l"):
                                            out_payload = (
                                                guarded[:-8] + footer + b"\x1b[?2026l"
                                            )
                                        else:
                                            out_payload = guarded + footer
                                        _write_all(tty_out_fd, out_payload)
                                        footer_drawn = True
                                        if not in_alt_screen[0]:
                                            normal_footer_state[0] = (
                                                rows,
                                                self._last_footer_vis_lens[0],
                                                self._last_footer_vis_lens[1],
                                            )
                                        last_bar_paint = now
                                        state_dirty[0] = False
                                        pending_region_restore = False
                                else:
                                    _write_all(tty_out_fd, guarded)
                        else:
                            prepared = erase_normal_footer_before_alt_screen(
                                complete,
                                in_alt_screen=in_alt_screen,
                                normal_footer_state=normal_footer_state,
                            )
                            _write_all(tty_out_fd, prepared)

                # Periodic/idle repaint when background router finishes classifying a step
                # or once an animated terminal resize settles
                now = time.monotonic()
                resize_settling = resize_flag[0] or (
                    resize_settle_deadline > 0.0 and now < resize_settle_deadline
                )
                if not resize_settling and resize_settle_deadline > 0.0:
                    resize_settle_deadline = 0.0

                if (
                    use_footer
                    and not in_sync_update
                    and not resize_settling
                    and not pending_child_bytes
                    and (
                        pending_region_restore
                        or not footer_drawn
                        or (
                            state_dirty[0]
                            and (now - last_bar_paint) >= self._min_refresh_interval
                        )
                    )
                ):
                    cur_rows, cur_cols = get_terminal_size(tty_out_fd)
                    if (cur_rows, cur_cols) != (rows, cols):
                        resize_flag[0] = True
                    else:
                        _write_all(
                            tty_out_fd,
                            self._build_footer_sequence(
                                rows,
                                cols,
                                reapply_scroll_region=(
                                    pending_region_restore or not footer_drawn
                                ),
                                scroll_region=active_scroll_region[0],
                            ),
                        )
                        footer_drawn = True
                        if not in_alt_screen[0]:
                            normal_footer_state[0] = (
                                rows,
                                self._last_footer_vis_lens[0],
                                self._last_footer_vis_lens[1],
                            )
                        last_bar_paint = now
                        state_dirty[0] = False
                        pending_region_restore = False

                if proc.poll() is not None and master_fd not in rlist:
                    # Drain any remaining bytes on master_fd before exiting loop
                    while True:
                        try:
                            ready, _, _ = select.select([master_fd], [], [], 0.02)
                            if not ready:
                                break
                            tail = os.read(master_fd, 16384)
                            if not tail:
                                break
                            _track_sync_mode(tail, time.monotonic())
                            tail_out = erase_normal_footer_before_alt_screen(
                                tail,
                                in_alt_screen=in_alt_screen,
                                normal_footer_state=normal_footer_state,
                            )
                            _write_all(tty_out_fd, tail_out)
                        except OSError:
                            break
                    break

        finally:
            self._state.set_on_change(None)
            signal.signal(signal.SIGWINCH, prev_sigwinch)
            try:
                termios.tcsetattr(stdin_fd, termios.TCSADRAIN, orig_termios)
            except termios.error:
                pass

            if in_sync_update:
                _write_all(tty_out_fd, b"\x1b[?2026l")
            if in_alt_screen[0]:
                _write_all(
                    tty_out_fd,
                    b"\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?25h\x1b[?1049l",
                )

            cur_exit_rows, _ = get_terminal_size(tty_out_fd)
            rows_to_clear: set[int] = set()
            if use_footer:
                rows_to_clear.add(max(1, cur_exit_rows - 1))
                rows_to_clear.add(max(1, cur_exit_rows))
            if normal_footer_state[0] is not None:
                nf_rows, _, _ = normal_footer_state[0]
                if nf_rows - 1 <= cur_exit_rows:
                    rows_to_clear.add(max(1, nf_rows - 1))
                if nf_rows <= cur_exit_rows:
                    rows_to_clear.add(max(1, nf_rows))

            if rows_to_clear:
                clear_cmds = "".join(
                    f"\x1b[{r_idx};1H\x1b[2K" for r_idx in sorted(rows_to_clear)
                )
                cleanup_seq = (
                    "\x1b7\x1b[0m"
                    f"{clear_cmds}"
                    "\x1b[r"
                    "\x1b8\r\n"
                ).encode("utf-8")
                _write_all(tty_out_fd, cleanup_seq)

            try:
                os.close(master_fd)
            except OSError:
                pass
            os.close(tty_out_fd)
            os.close(wakeup_r)
            os.close(wakeup_w)

        return int(proc.wait() or 0)
