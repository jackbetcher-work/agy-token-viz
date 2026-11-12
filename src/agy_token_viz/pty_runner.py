"""PTY terminal wrapper reserving a 2-line bottom status bar while running `agy` cleanly."""

from __future__ import annotations

import errno
import fcntl
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
)
from agy_token_viz.state import SessionTokenState


BAR_ROWS = 2
MIN_TERMINAL_ROWS_FOR_FOOTER = 6

# Detect child sequences that reset scroll margins or clear/switch screen buffers
_REGION_OR_CLEAR_RE = re.compile(
    rb"\x1b\[(?:\?(?:1049|1047|47)[hl]|[012]?J|[0-9;]*r)"
)

# Track alternate screen buffer entry ('h') and exit ('l')
_ALT_BUFFER_RE = re.compile(rb"\x1b\[\?(?:1049|1047|47)([hl])")

# Child sequences that exit the alternate screen buffer or reset DECSTBM scroll margins
_MARGIN_RESTORE_RE = re.compile(rb"\x1b\[(?:\?(?:1049|1047|47)l|[0-9;]*r)")


def inject_scroll_region_guards(data: bytes, child_rows: int) -> bytes:
    """
    Immediately re-apply `DECSTBM` (`\\x1b7\\x1b[1;{child_rows}r\\x1b8`) right after any
    sequence in `data` that exits the alternate screen buffer (`\\x1b[?1049l`) or resets
    scroll margins (`\\x1b[...r`), before subsequent newlines in the same chunk can scroll
    into the reserved bottom footer rows.
    """
    if not data or not _MARGIN_RESTORE_RE.search(data):
        return data
    guard = f"\x1b7\x1b[1;{max(1, child_rows)}r\x1b8".encode("ascii")
    return _MARGIN_RESTORE_RE.sub(lambda m: m.group(0) + guard, data)


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


def set_pty_winsize(fd: int, rows: int, cols: int) -> None:
    """Set `(rows, cols)` on a PTY descriptor (triggers SIGWINCH to the child group)."""
    try:
        packed = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


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
    ) -> None:
        self._cmd = list(cmd)
        self._state = state
        self._use_color = should_use_color() if use_color is None else use_color
        self._min_refresh_interval = min_refresh_interval

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

        self._state.set_on_change(_notify)
        if on_before_spawn is not None:
            on_before_spawn()

        proc = subprocess.Popen(self._cmd)
        if on_process_spawn is not None:
            on_process_spawn(proc.pid)
        drew_stderr_bar = False

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
                    snap = self._state.snapshot()
                    line = render_single_status_line(
                        snap, cols=cols, use_color=self._use_color
                    )
                    _write_all(stderr_fd, f"\r\x1b[2K{line}".encode("utf-8"))
                    drew_stderr_bar = True
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            rc = proc.wait()
        finally:
            self._state.set_on_change(None)
            if drew_stderr_bar:
                _write_all(stderr_fd, b"\r\x1b[2K")
            os.close(stderr_fd)
            os.close(wakeup_r)
            os.close(wakeup_w)

        return int(proc.returncode or 0)

    def _build_footer_sequence(
        self,
        rows: int,
        cols: int,
        reapply_scroll_region: bool = False,
    ) -> bytes:
        child_rows = max(1, rows - BAR_ROWS)
        snap = self._state.snapshot()
        line1, line2 = render_status_lines(snap, cols=cols, use_color=self._use_color)

        # IMPORTANT: Save child cursor (\x1b7) BEFORE DECSTBM (\x1b[1;{child_rows}r),
        # because DECSTBM homes the cursor to (1, 1) per VT100/xterm specification.
        parts: list[str] = ["\x1b7"]
        if reapply_scroll_region:
            parts.append(f"\x1b[1;{child_rows}r")
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
        in_alt_screen = False

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

        proc = subprocess.Popen(
            self._cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        if on_process_spawn is not None:
            on_process_spawn(proc.pid)

        pending_child_bytes = b""
        last_bar_paint = time.monotonic()

        try:
            tty.setraw(stdin_fd)

            while True:
                timeout = 0.05 if state_dirty[0] else 0.25
                try:
                    rlist, _, _ = select.select(
                        [master_fd, stdin_fd, wakeup_r], [], [], timeout
                    )
                except InterruptedError:
                    rlist = []

                if wakeup_r in rlist:
                    try:
                        os.read(wakeup_r, 1024)
                    except OSError:
                        pass

                if resize_flag[0]:
                    resize_flag[0] = False
                    rows, cols = get_terminal_size(tty_out_fd)
                    use_footer = rows >= MIN_TERMINAL_ROWS_FOR_FOOTER
                    child_rows = max(1, rows - BAR_ROWS) if use_footer else rows
                    set_pty_winsize(master_fd, child_rows, cols)
                    if use_footer and not in_alt_screen:
                        _write_all(
                            tty_out_fd,
                            self._build_footer_sequence(
                                rows, cols, reapply_scroll_region=True
                            ),
                        )
                        last_bar_paint = time.monotonic()
                        state_dirty[0] = False

                if stdin_fd in rlist:
                    try:
                        user_bytes = os.read(stdin_fd, 4096)
                    except OSError:
                        user_bytes = b""
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
                            _write_all(tty_out_fd, pending_child_bytes)
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
                        if use_footer:
                            for match in _ALT_BUFFER_RE.finditer(complete):
                                in_alt_screen = match.group(1) == b"h"
                            guarded = inject_scroll_region_guards(complete, child_rows)
                            if not in_alt_screen:
                                needs_region = bool(_REGION_OR_CLEAR_RE.search(complete))
                                now = time.monotonic()
                                if needs_region or (
                                    state_dirty[0]
                                    and (now - last_bar_paint) >= self._min_refresh_interval
                                ):
                                    footer = self._build_footer_sequence(
                                        rows,
                                        cols,
                                        reapply_scroll_region=needs_region,
                                    )
                                    _write_all(tty_out_fd, guarded + footer)
                                    last_bar_paint = now
                                    state_dirty[0] = False
                                else:
                                    _write_all(tty_out_fd, guarded)
                            else:
                                _write_all(tty_out_fd, guarded)
                        else:
                            _write_all(tty_out_fd, complete)

                # Periodic/idle repaint when background router finishes classifying a step
                now = time.monotonic()
                if (
                    use_footer
                    and not in_alt_screen
                    and state_dirty[0]
                    and not pending_child_bytes
                    and (now - last_bar_paint) >= self._min_refresh_interval
                ):
                    _write_all(
                        tty_out_fd,
                        self._build_footer_sequence(
                            rows, cols, reapply_scroll_region=False
                        ),
                    )
                    last_bar_paint = now
                    state_dirty[0] = False

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
                            _write_all(tty_out_fd, tail)
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

            if use_footer:
                rows, _ = get_terminal_size(tty_out_fd)
                # Clear the 2 pinned footer lines, reset DECSTBM margins, place cursor cleanly
                cleanup_seq = (
                    "\x1b7"
                    f"\x1b[{max(1, rows - 1)};1H\x1b[2K"
                    f"\x1b[{max(1, rows)};1H\x1b[2K"
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
