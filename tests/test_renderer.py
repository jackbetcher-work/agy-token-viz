"""Unit tests for tqdm-style 2-line status bar and exit summary rendering (`renderer.py`)."""

from __future__ import annotations

import pytest

from agy_token_viz.renderer import (
    allocate_bar_widths,
    char_display_width,
    format_token_count,
    render_final_summary,
    render_status_lines,
    truncate_visible,
    visible_len,
)
from agy_token_viz.state import SessionTokenState, StepRecord


def test_format_token_count() -> None:
    assert format_token_count(0) == "0"
    assert format_token_count(842) == "842"
    assert format_token_count(8431) == "8.4k"
    assert format_token_count(125_400) == "125k"
    assert format_token_count(1_500_000) == "1.50M"


@pytest.mark.parametrize("cols", [40, 65, 80, 100, 140])
@pytest.mark.parametrize("use_color", [True, False])
def test_render_status_lines_never_exceeds_terminal_width(cols: int, use_color: bool) -> None:
    state = SessionTokenState()
    state.set_router_status("qwen2.5-0.5b")
    state.record_classified_step(
        StepRecord("c1", 1, input_tokens=8000, cache_read_tokens=20000, output_tokens=600),
        "Exploration",
    )
    state.record_classified_step(
        StepRecord("c1", 2, input_tokens=12000, cache_read_tokens=28000, output_tokens=1400),
        "Coding",
    )
    state.record_classified_step(
        StepRecord("c1", 3, input_tokens=4000, cache_read_tokens=40000, output_tokens=500),
        "Testing",
    )

    snap = state.snapshot()
    line1, line2 = render_status_lines(snap, cols=cols, use_color=use_color)

    assert visible_len(line1) <= cols
    assert visible_len(line2) <= cols
    assert "Tokens" in line1


def test_allocate_bar_widths_sums_to_exact_bar_width() -> None:
    state = SessionTokenState(categories=("A", "B", "C"))
    state.record_classified_step(StepRecord("c1", 1, 100, 0, 0), "A")
    state.record_classified_step(StepRecord("c1", 2, 100, 100, 0), "B")
    state.record_classified_step(StepRecord("c1", 3, 100, 200, 0), "C")
    snap = state.snapshot()

    widths = allocate_bar_widths(snap.categories, 20)
    assert sum(widths) == 20
    assert sorted(widths) == [6, 7, 7]


def test_render_final_summary_contains_categories_and_billed_note() -> None:
    state = SessionTokenState()
    state.record_classified_step(
        StepRecord("c1", 1, input_tokens=3000, cache_read_tokens=0, output_tokens=400),
        "Planning",
    )
    # Step 2 re-sends 1000 uncached tail tokens from Step 1 (cache=2400, in=1200 -> prompt=3600, net_in=200)
    state.record_classified_step(
        StepRecord("c1", 2, input_tokens=1200, cache_read_tokens=2400, output_tokens=100),
        "Communication",
    )
    summary = render_final_summary(state.snapshot(), use_color=False)
    assert "Session Token Summary" in summary
    assert "In: 3.2k (Billed: 4.2k)" in summary
    assert "Planning" in summary
    assert "3.4k" in summary


def test_render_status_and_summary_surface_thinking_tokens() -> None:
    state = SessionTokenState()
    state.set_router_status("qwen2.5-3b")
    state.record_classified_step(
        StepRecord(
            "c1",
            1,
            input_tokens=4000,
            cache_read_tokens=0,
            output_tokens=1500,
            thinking_tokens=1200,
        ),
        "Exploration",
    )
    snap = state.snapshot()
    line1, _ = render_status_lines(snap, cols=120, use_color=False)
    assert "Think:1.2k" in line1

    summary = render_final_summary(snap, use_color=False)
    assert "Out: 1.5k (Think: 1.2k)" in summary
    assert "think:1.2k" in summary


@pytest.mark.parametrize("cols", [1, 5, 12, 19])
@pytest.mark.parametrize("use_color", [True, False])
def test_render_status_lines_ultra_narrow_split_pane_never_wraps(
    cols: int, use_color: bool
) -> None:
    state = SessionTokenState()
    state.record_classified_step(
        StepRecord("c1", 1, input_tokens=4000, cache_read_tokens=8000, output_tokens=500),
        "Exploration",
    )
    line1, line2 = render_status_lines(state.snapshot(), cols=cols, use_color=use_color)
    assert visible_len(line1) <= cols
    assert visible_len(line2) <= cols


@pytest.mark.parametrize("cols", [10, 24, 42, 80])
@pytest.mark.parametrize("use_color", [True, False])
def test_render_status_lines_wide_cjk_and_emoji_categories_fit_columns(
    cols: int, use_color: bool
) -> None:
    assert char_display_width("探") == 2
    assert char_display_width("🚀") == 2
    assert char_display_width("\u0301") == 0
    # A 2-column glyph must not overflow a 1-column budget
    assert truncate_visible("🚀Deploy", 1, use_color=False) == ""
    assert visible_len(truncate_visible("探索中", 3, use_color=False)) == 2

    state = SessionTokenState(categories=("探索調査", "🚀Deploy", "e\u0301dit"))
    state.record_classified_step(StepRecord("c1", 1, 2000, 0, 300), "探索調査")
    state.record_classified_step(StepRecord("c1", 2, 1500, 2000, 200), "🚀Deploy")
    state.record_classified_step(StepRecord("c1", 3, 500, 3500, 100), "e\u0301dit")

    line1, line2 = render_status_lines(state.snapshot(), cols=cols, use_color=use_color)
    assert visible_len(line1) <= cols
    assert visible_len(line2) <= cols

