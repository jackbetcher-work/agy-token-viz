"""Width-adaptive tqdm-style 2-line terminal status bar and exit summary renderer."""

from __future__ import annotations

import os
import re
from typing import Sequence

from agy_token_viz.config import SHORT_CATEGORY_LABELS
from agy_token_viz.state import CategorySnapshot, LedgerSnapshot


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

# Distinct terminal-theme-friendly ANSI colors & block characters per category slot
CATEGORY_STYLES: tuple[tuple[str, str], ...] = (
    ("\x1b[36m", "█"),  # Cyan
    ("\x1b[34m", "▓"),  # Blue
    ("\x1b[32m", "▒"),  # Green
    ("\x1b[33m", "░"),  # Yellow
    ("\x1b[35m", "▪"),  # Magenta
    ("\x1b[96m", "▫"),  # Bright Cyan
    ("\x1b[92m", "◆"),  # Bright Green
)


def should_use_color(env: dict[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    if source.get("NO_COLOR"):
        return False
    if source.get("TERM", "").lower() == "dumb":
        return False
    return True


def visible_len(text: str) -> int:
    """Return display character count excluding ANSI CSI escape sequences."""
    return len(_ANSI_ESCAPE_RE.sub("", text))


def truncate_visible(text: str, max_cols: int, use_color: bool = True) -> str:
    """Truncate a string containing ANSI codes so its visible length never exceeds max_cols."""
    if max_cols <= 0:
        return ""
    if visible_len(text) <= max_cols:
        return text

    out: list[str] = []
    vis = 0
    idx = 0
    n = len(text)
    while idx < n and vis < max_cols:
        match = _ANSI_ESCAPE_RE.match(text, idx)
        if match:
            out.append(match.group(0))
            idx = match.end()
            continue
        out.append(text[idx])
        vis += 1
        idx += 1

    if use_color:
        out.append(RESET)
    return "".join(out)


def format_token_count(count: int) -> str:
    """Format token counts compactly (e.g., 842, 12.4k, 148k, 1.25M)."""
    n = max(0, int(count))
    if n < 1_000:
        return str(n)
    if n < 100_000:
        val = n / 1_000.0
        return f"{val:.1f}k" if val < 10 or (n % 1000 >= 100) else f"{val:.1f}k"
    if n < 1_000_000:
        return f"{n / 1_000.0:.0f}k"
    return f"{n / 1_000_000.0:.2f}M"


def short_label(category_name: str) -> str:
    if category_name in SHORT_CATEGORY_LABELS:
        return SHORT_CATEGORY_LABELS[category_name]
    if "/" in category_name:
        return category_name.split("/", 1)[0][:8]
    return category_name[:10]


def allocate_bar_widths(
    categories: Sequence[CategorySnapshot],
    bar_width: int,
) -> list[int]:
    """Distribute `bar_width` character cells proportionally using largest-remainder rounding."""
    if bar_width <= 0 or not categories:
        return [0 for _ in categories]

    total = sum(c.active_tokens for c in categories)
    if total <= 0:
        return [0 for _ in categories]

    exact = [(c.active_tokens / total) * bar_width for c in categories]
    floors = [int(x) for x in exact]
    remainder = bar_width - sum(floors)

    ranked_indices = sorted(
        range(len(categories)),
        key=lambda i: (exact[i] - floors[i], categories[i].active_tokens),
        reverse=True,
    )
    for i in ranked_indices[:remainder]:
        floors[i] += 1
    return floors


def _render_segmented_bar(
    categories: Sequence[CategorySnapshot],
    bar_width: int,
    use_color: bool = True,
) -> str:
    widths = allocate_bar_widths(categories, bar_width)
    if sum(widths) == 0:
        empty = "·" * max(1, bar_width)
        return f"│{DIM}{empty}{RESET}│" if use_color else f"|{empty}|"

    segments: list[str] = []
    for idx, width in enumerate(widths):
        if width <= 0:
            continue
        color, glyph = CATEGORY_STYLES[idx % len(CATEGORY_STYLES)]
        block = glyph * width
        if use_color:
            segments.append(f"{color}{block}{RESET}")
        else:
            segments.append(block)
    left_sep = "│" if use_color else "|"
    right_sep = "│" if use_color else "|"
    return f"{left_sep}{''.join(segments)}{right_sep}"


def render_status_lines(
    snapshot: LedgerSnapshot,
    cols: int = 100,
    use_color: bool = True,
) -> tuple[str, str]:
    """
    Render the 2-line tqdm-style pinned terminal status bar adapted to `cols` columns.

    Line 1: Total active tokens + proportional category bar + In/Cache/Out + step/router status
    Line 2: Per-category running tally with token counts and percentages
    """
    width = max(20, int(cols))

    total_str = format_token_count(snapshot.total_active_tokens)
    in_str = format_token_count(snapshot.total_input_tokens)
    cache_str = format_token_count(snapshot.total_cache_read_tokens)
    out_str = format_token_count(snapshot.total_output_tokens)
    has_think = snapshot.total_thinking_tokens > 0
    think_str = format_token_count(snapshot.total_thinking_tokens)

    pending_note = f"+{snapshot.pending_steps}q" if snapshot.pending_steps > 0 else ""
    steps_str = f"{snapshot.classified_steps}{pending_note}"

    prefix_plain = f"Tokens {total_str:>5} "
    if use_color:
        prefix = f"{BOLD}Tokens{RESET} {BOLD}{total_str:>5}{RESET} "
    else:
        prefix = prefix_plain

    think_full_plain = f" Think:{think_str}" if has_think else ""
    think_med_plain = f" T:{think_str}" if has_think else ""
    think_full_color = f" {DIM}Think:{RESET}{think_str}" if has_think else ""
    think_med_color = f" {DIM}T:{RESET}{think_str}" if has_think else ""

    # Adapt suffix verbosity to terminal width so the tqdm bar stays readable
    full_suffix_plain = (
        f" In:{in_str} Cache:{cache_str} Out:{out_str}{think_full_plain} "
        f"[{steps_str} steps | {snapshot.router_status}]"
    )
    med_suffix_plain = (
        f" In:{in_str} C:{cache_str} Out:{out_str}{think_med_plain} "
        f"[{steps_str}st|{snapshot.router_status}]"
    )
    compact_suffix_plain = f" I:{in_str} C:{cache_str} O:{out_str}"

    if width - len(prefix_plain) - len(full_suffix_plain) >= 14:
        suffix_plain = full_suffix_plain
        if use_color:
            suffix = (
                f" {DIM}In:{RESET}{in_str} {DIM}Cache:{RESET}{cache_str} "
                f"{DIM}Out:{RESET}{out_str}{think_full_color} "
                f"{DIM}[{steps_str} steps | {snapshot.router_status}]{RESET}"
            )
        else:
            suffix = full_suffix_plain
    elif width - len(prefix_plain) - len(med_suffix_plain) >= 12:
        suffix_plain = med_suffix_plain
        if use_color:
            suffix = (
                f" {DIM}In:{RESET}{in_str} {DIM}C:{RESET}{cache_str} "
                f"{DIM}Out:{RESET}{out_str}{think_med_color} "
                f"{DIM}[{steps_str}st|{snapshot.router_status}]{RESET}"
            )
        else:
            suffix = med_suffix_plain
    else:
        suffix_plain = compact_suffix_plain
        suffix = compact_suffix_plain

    avail_for_bar = width - len(prefix_plain) - len(suffix_plain) - 2
    bar_width = max(6, min(32, avail_for_bar))
    bar = _render_segmented_bar(snapshot.categories, bar_width, use_color=use_color)
    line1 = truncate_visible(f"{prefix}{bar}{suffix}", width, use_color=use_color)

    # Line 2: Category breakdown
    wide_items: list[str] = []
    med_items: list[str] = []
    narrow_items: list[str] = []

    for idx, cat in enumerate(snapshot.categories):
        color, glyph = CATEGORY_STYLES[idx % len(CATEGORY_STYLES)]
        lbl = short_label(cat.name)
        tok = format_token_count(cat.active_tokens)
        pct = f"{cat.percentage:.0f}%"
        marker = f"{color}{glyph}{RESET}" if use_color else glyph

        if use_color:
            wide_items.append(f"{marker} {BOLD}{lbl}{RESET} {tok} {DIM}({pct}){RESET}")
            med_items.append(f"{marker}{lbl}:{tok}({pct})")
            narrow_items.append(f"{color}{lbl}{RESET}:{pct}")
        else:
            wide_items.append(f"{glyph} {lbl} {tok} ({pct})")
            med_items.append(f"{glyph}{lbl}:{tok}({pct})")
            narrow_items.append(f"{lbl}:{pct}")

    line2_wide = "  ".join(wide_items)
    line2_med = " ".join(med_items)
    line2_narrow = " ".join(narrow_items)

    if visible_len(line2_wide) <= width:
        line2 = line2_wide
    elif visible_len(line2_med) <= width:
        line2 = line2_med
    else:
        line2 = truncate_visible(line2_narrow, width, use_color=use_color)

    return line1, line2


def render_single_status_line(
    snapshot: LedgerSnapshot,
    cols: int = 100,
    use_color: bool = True,
) -> str:
    """Single-line tqdm status bar used when stdout is redirected/piped and stderr is a TTY."""
    line1, _ = render_status_lines(snapshot, cols=cols, use_color=use_color)
    return line1


def render_final_summary(
    snapshot: LedgerSnapshot,
    use_color: bool = True,
) -> str:
    """Render a concise post-session token breakdown by activity category."""
    if snapshot.total_steps == 0:
        header = "[agy-token-viz] No agent steps recorded in this session."
        return f"{DIM}{header}{RESET}" if use_color else header

    lines: list[str] = []
    in_summary = f"In: {format_token_count(snapshot.total_input_tokens)}"
    if snapshot.total_billed_input_tokens != snapshot.total_input_tokens:
        in_summary += f" (Billed: {format_token_count(snapshot.total_billed_input_tokens)})"

    out_summary = f"Out: {format_token_count(snapshot.total_output_tokens)}"
    if snapshot.total_thinking_tokens > 0:
        out_summary += f" (Think: {format_token_count(snapshot.total_thinking_tokens)})"

    title = (
        f"[agy-token-viz] Session Token Summary "
        f"(Active: {format_token_count(snapshot.total_active_tokens)} | "
        f"{in_summary} | "
        f"Cache: {format_token_count(snapshot.total_cache_read_tokens)} | "
        f"{out_summary} | "
        f"Steps: {snapshot.total_steps})"
    )
    lines.append(f"{BOLD}{title}{RESET}" if use_color else title)

    max_name = max((len(c.name) for c in snapshot.categories), default=12)
    bar_cells = 20
    for idx, cat in enumerate(snapshot.categories):
        color, glyph = CATEGORY_STYLES[idx % len(CATEGORY_STYLES)]
        filled = int(round((cat.percentage / 100.0) * bar_cells))
        filled = max(0, min(bar_cells, filled))
        empty = bar_cells - filled
        think_detail = (
            f" think:{format_token_count(cat.thinking_tokens)}"
            if cat.thinking_tokens > 0
            else ""
        )
        if use_color:
            mini_bar = f"│{color}{glyph * filled}{RESET}{DIM}{'·' * empty}{RESET}│"
            row = (
                f"  {cat.name:<{max_name}}  {mini_bar} "
                f"{format_token_count(cat.active_tokens):>6} "
                f"({cat.percentage:5.1f}%)  "
                f"{DIM}[{cat.steps} step{'s' if cat.steps != 1 else ''}, "
                f"in:{format_token_count(cat.input_tokens)} out:{format_token_count(cat.output_tokens)}{think_detail}]{RESET}"
            )
        else:
            mini_bar = f"|{glyph * filled}{'·' * empty}|"
            row = (
                f"  {cat.name:<{max_name}}  {mini_bar} "
                f"{format_token_count(cat.active_tokens):>6} "
                f"({cat.percentage:5.1f}%)  "
                f"[{cat.steps} step{'s' if cat.steps != 1 else ''}, "
                f"in:{format_token_count(cat.input_tokens)} out:{format_token_count(cat.output_tokens)}{think_detail}]"
            )
        lines.append(row)

    return "\n".join(lines)
