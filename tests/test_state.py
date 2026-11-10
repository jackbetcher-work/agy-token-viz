"""Unit tests for in-memory token accounting state (`state.py`)."""

from __future__ import annotations

import threading

from agy_token_viz.state import Category, SessionTokenState, StepRecord, ToolCallSummary
from agy_token_viz.tokenizer import count_tokens_locally


def test_register_and_classify_step_updates_buckets() -> None:
    changes: list[int] = []
    state = SessionTokenState(
        categories=("Planning", "Exploration", "Coding"),
        on_change=lambda: changes.append(1),
    )

    step = StepRecord(
        conversation_id="conv-1",
        step_index=1,
        input_tokens=4000,
        cache_read_tokens=12000,
        output_tokens=500,
        thinking="Let's inspect the directory layout.",
    )
    assert state.register_step(step) is True
    # Duplicate registration of the same (conversation_id, step_index) is ignored
    assert state.register_step(step) is False

    mid_snap = state.snapshot()
    assert mid_snap.total_active_tokens == 4500
    assert mid_snap.unclassified_active_tokens == 4500
    assert mid_snap.classified_active_tokens == 0
    assert mid_snap.total_cache_read_tokens == 12000
    assert mid_snap.pending_steps == 1

    assigned = state.complete_classification("conv-1", 1, "exploration")
    assert assigned == "Exploration"

    final_snap = state.snapshot()
    assert final_snap.unclassified_active_tokens == 0
    assert final_snap.classified_active_tokens == 4500
    assert final_snap.pending_steps == 0
    assert final_snap.classified_steps == 1
    assert final_snap.last_category == "Exploration"

    by_name = {c.name: c for c in final_snap.categories}
    assert by_name["Exploration"].active_tokens == 4500
    assert by_name["Exploration"].input_tokens == 4000
    assert by_name["Exploration"].output_tokens == 500
    assert by_name["Exploration"].cache_read_tokens == 12000
    assert by_name["Exploration"].percentage == 100.0
    assert by_name["Planning"].active_tokens == 0
    assert len(changes) >= 2


def test_net_context_delta_eliminates_cache_tail_and_output_double_counting() -> None:
    """
    Verify:
    1. Prefix-cache block-tail re-reads are NOT double-counted in `total_input_tokens`.
    2. Step N's `output_tokens` re-entering history on Step N+1 are NOT counted twice.
    3. Tool-result prompt growth on Step N+1 is attributed back to Step N's category.
    """
    state = SessionTokenState(categories=("Exploration", "Communication"))

    # Step 1: System + user prompt (8000 in, 0 cache), calls read_file (300 out) -> end_context = 8300
    s1 = StepRecord(
        conversation_id="c1",
        step_index=1,
        input_tokens=8000,
        cache_read_tokens=0,
        output_tokens=300,
        tool_calls=(ToolCallSummary("read_file", "big_module.py"),),
        preceded_by_user_input=True,
    )
    state.record_classified_step(s1, "Exploration")

    # Step 2: File content adds 10,000 tokens -> total prompt = 18,300.
    # Suppose prefix cache only cached 6,000 of Step 1's 8,000 prompt, so API bills 12,300 in!
    # Net new input must be 18,300 - 8,300 = 10,000 (attributed back to Step 1's Exploration!),
    # while Step 2 (Communication reply, 1,200 out) only gets charged its 1,200 output tokens.
    s2 = StepRecord(
        conversation_id="c1",
        step_index=2,
        input_tokens=12300,
        cache_read_tokens=6000,
        output_tokens=1200,
        content="Here is how big_module.py works...",
        preceded_by_user_input=False,
    )
    state.record_classified_step(s2, "Communication")

    # Step 3: User sends a 150-token follow-up prompt.
    # Step 2's 1,200 output tokens haven't been cached yet, so API bills 1,350 in + 18,300 cache = 19,650 prompt.
    # Net new input must be 19,650 - (18,300 + 1,200) = 150 (staying on Step 3 since preceded_by_user_input=True).
    s3 = StepRecord(
        conversation_id="c1",
        step_index=3,
        input_tokens=1350,
        cache_read_tokens=18300,
        output_tokens=200,
        content="Short answer.",
        preceded_by_user_input=True,
    )
    state.record_classified_step(s3, "Communication")

    snap = state.snapshot()
    by_name = {c.name: c for c in snap.categories}

    # Total context at end of Step 3 = 19,650 + 200 = 19,850 == total_active_tokens!
    assert snap.total_input_tokens == 8000 + 10000 + 150  # 18,150 net input
    assert snap.total_billed_input_tokens == 8000 + 12300 + 1350  # 21,650 raw billed input
    assert snap.total_output_tokens == 300 + 1200 + 200  # 1,700 output
    assert snap.total_active_tokens == 19850

    assert by_name["Exploration"].input_tokens == 18000
    assert by_name["Exploration"].output_tokens == 300
    assert by_name["Exploration"].active_tokens == 18300

    assert by_name["Communication"].input_tokens == 150
    assert by_name["Communication"].output_tokens == 1400
    assert by_name["Communication"].active_tokens == 1550


def test_tool_result_attribution_when_previous_step_is_still_pending() -> None:
    """Verify tool-result delta lands properly even if Step 1 has not finished LLM classification yet."""
    state = SessionTokenState(categories=("Exploration", "Communication"))

    s1 = StepRecord(
        "c1",
        1,
        input_tokens=1000,
        cache_read_tokens=0,
        output_tokens=100,
        tool_calls=(ToolCallSummary("read_file", "a.py"),),
    )
    s2 = StepRecord(
        "c1",
        2,
        input_tokens=2600,
        cache_read_tokens=1000,
        output_tokens=400,
        content="Summary",
    )

    # Both steps arrive before classifier finishes Step 1
    state.register_step(s1)
    state.register_step(s2)

    mid = state.snapshot()
    # Prompt_2 = 3600, prev_end = 1100 -> net_in_2 = 2500. Total active = 1000 + 100 + 2500 + 400 = 4000
    assert mid.unclassified_active_tokens == 4000

    state.complete_classification("c1", 1, "Exploration")
    state.complete_classification("c1", 2, "Communication")

    final = state.snapshot()
    by_name = {c.name: c for c in final.categories}
    assert final.unclassified_active_tokens == 0
    assert by_name["Exploration"].active_tokens == 3600  # 1000 + 2500 + 100
    assert by_name["Communication"].active_tokens == 400


def test_context_compaction_resets_baseline_cleanly() -> None:
    state = SessionTokenState(categories=("Exploration", "Communication"))
    state.record_classified_step(
        StepRecord("c1", 1, input_tokens=50000, cache_read_tokens=0, output_tokens=1000),
        "Exploration",
    )
    # Simulate `/compact` shrinking prompt from 51,000 down to 6,000
    state.record_classified_step(
        StepRecord("c1", 2, input_tokens=6000, cache_read_tokens=0, output_tokens=300, preceded_by_user_input=True),
        "Communication",
    )
    # Next turn grows from 6,300 to 7,000 -> net_in = 700
    state.record_classified_step(
        StepRecord("c1", 3, input_tokens=1000, cache_read_tokens=6000, output_tokens=200, preceded_by_user_input=True),
        "Communication",
    )
    snap = state.snapshot()
    assert snap.total_input_tokens == 50000 + 0 + 700
    assert snap.total_output_tokens == 1000 + 300 + 200


def test_thread_safe_concurrent_classifications() -> None:
    state = SessionTokenState(categories=("Planning", "Coding"))

    def _worker(conv_prefix: str, category: str) -> None:
        for offset in range(25):
            rec = StepRecord(
                conversation_id=f"{conv_prefix}-{offset}",
                step_index=1,
                input_tokens=100,
                cache_read_tokens=50,
                output_tokens=20,
            )
            state.record_classified_step(rec, category)

    t1 = threading.Thread(target=_worker, args=("plan-conv", "Planning"))
    t2 = threading.Thread(target=_worker, args=("code-conv", "Coding"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = state.snapshot()
    assert snap.total_steps == 50
    assert snap.classified_steps == 50
    assert snap.total_active_tokens == 50 * 120
    by_name = {c.name: c for c in snap.categories}
    assert by_name["Planning"].active_tokens == 25 * 120
    assert by_name["Coding"].active_tokens == 25 * 120
    assert abs(by_name["Planning"].percentage - 50.0) < 1e-6


def test_thinking_tokens_route_to_planning_and_preserve_stripped_turn_deltas() -> None:
    """
    Verify:
    1. A step's `thinking_tokens` are attributed to `Planning`, while `visible_output_tokens`
       and `net_input_tokens` go to the step's classified action category.
    2. When `agy`/Gemini strips prior thinking tokens on the next turn
       (`prev_visible_end_context <= prompt_tokens < prev_end_context`), new tool-result
       or message input tokens are still captured instead of clamping to 0.
    """
    state = SessionTokenState(categories=("Planning", "Exploration", "Communication"))

    # Step 1: 8,000 in, 2,500 out (2,400 thinking + 100 visible tool call)
    # prev_visible_end_context = 8,100; prev_end_context = 10,500
    s1 = StepRecord(
        conversation_id="c-think",
        step_index=1,
        input_tokens=8000,
        cache_read_tokens=0,
        output_tokens=2500,
        thinking="Summarizing thoughts...",
        tool_calls=(ToolCallSummary("read_file", "module.py"),),
        preceded_by_user_input=True,
        thinking_tokens=2400,
    )
    state.record_classified_step(s1, "Exploration")

    # Step 2: Harness strips Step 1's 2,400 thinking tokens from prompt history,
    # keeping 8,100 visible context + adding 900 tokens of `read_file` tool output -> prompt = 9,000.
    # Because 8,100 <= 9,000 < 10,500, net_in must be 9,000 - 8,100 = 900 (attributed to Step 1's Exploration!).
    s2 = StepRecord(
        conversation_id="c-think",
        step_index=2,
        input_tokens=1000,
        cache_read_tokens=8000,
        output_tokens=600,
        thinking="Preparing explanation...",
        content="Here is how module.py works.",
        preceded_by_user_input=False,
        thinking_tokens=200,
    )
    state.record_classified_step(s2, "Communication")

    snap = state.snapshot()
    by_name = {c.name: c for c in snap.categories}

    assert snap.total_thinking_tokens == 2400 + 200
    assert snap.total_output_tokens == 2500 + 600
    assert snap.total_input_tokens == 8000 + 900

    # Planning receives 2,400 + 200 = 2,600 thinking tokens
    assert by_name["Planning"].active_tokens == 2600
    assert by_name["Planning"].output_tokens == 2600
    assert by_name["Planning"].thinking_tokens == 2600

    # Exploration receives 8,000 initial input + 900 tool output + 100 visible tool-call output = 9,000
    assert by_name["Exploration"].input_tokens == 8900
    assert by_name["Exploration"].output_tokens == 100
    assert by_name["Exploration"].active_tokens == 9000

    # Communication receives 400 visible reply output tokens
    assert by_name["Communication"].output_tokens == 400
    assert by_name["Communication"].active_tokens == 400


def test_step_1_harness_overhead_isolation_and_pruned_turn_boundary_payload_tracking() -> None:
    """
    Verify Acceptance Criteria 2 & 3:
    - Step 1 (`cache_read_tokens == 0`) isolates ~8k `Harness Overhead` tokens from `user_prompt`.
    - When `agy` strips prior thinking blocks across a turn/error boundary so `prompt_tokens`
      drops below `prev_end_context` (and even below `prev_prompt_tokens`: P_k < P_{k-1}),
      `SessionTokenState` directly tokenizes the newly appended payload via SentencePiece.
    """
    state = SessionTokenState()

    initial_query = "<USER_REQUEST>\nExplain the architecture of this repository\n</USER_REQUEST>"
    user_query_tokens = count_tokens_locally(initial_query)
    step_1_total_in = 8046
    expected_harness_tokens = step_1_total_in - user_query_tokens

    # Step 1: 8,046 input tokens (7,900+ harness preamble + user query), 5,000 output tokens (4,900 thinking + 100 tool)
    s1 = StepRecord(
        conversation_id="c-harness",
        step_index=1,
        input_tokens=step_1_total_in,
        cache_read_tokens=0,
        output_tokens=5000,
        thinking="Planning initial repo inspection.",
        tool_calls=(ToolCallSummary("run_command", "ls -la"),),
        preceded_by_user_input=True,
        thinking_tokens=4900,
        user_prompt=initial_query,
    )
    state.record_classified_step(s1, Category.EXPLORATION)

    # Step 2: Harness strips all 4,900 thinking tokens from Step 1 AND prunes an earlier block so
    # total prompt (8,400) is smaller than Step 1's prev_end_context (13,046).
    # Meanwhile, `ls -la` returned a tool result payload.
    tool_output_text = "total 32\ndrwxr-xr-x 10 user staff 320 Nov 10 src\n-rw-r--r-- 1 user staff 641 pyproject.toml\n"
    expected_tool_tokens = count_tokens_locally(tool_output_text)

    s2 = StepRecord(
        conversation_id="c-harness",
        step_index=2,
        input_tokens=2400,
        cache_read_tokens=6000,  # P_2 = 8,400 < prev_end_context (13,046)
        output_tokens=300,
        content="Here is the directory breakdown.",
        preceded_by_user_input=False,
        thinking_tokens=0,
        appended_payload=tool_output_text,
    )
    state.record_classified_step(s2, Category.COMMUNICATION)

    snap = state.snapshot()
    by_name = {c.name: c for c in snap.categories}

    # Verify ~7.9k Harness Overhead is isolated in its dedicated category
    assert Category.HARNESS_OVERHEAD in by_name
    assert by_name[Category.HARNESS_OVERHEAD].active_tokens == expected_harness_tokens
    assert by_name[Category.HARNESS_OVERHEAD].input_tokens == expected_harness_tokens
    assert snap.total_harness_overhead_tokens == expected_harness_tokens
    assert 7000 <= expected_harness_tokens <= 9500

    # Exploration gets only `user_query_tokens` + `expected_tool_tokens` (from Step 2's tool result) + 100 visible tool call
    assert by_name[Category.EXPLORATION].input_tokens == user_query_tokens + expected_tool_tokens
    assert by_name[Category.EXPLORATION].output_tokens == 100
    assert by_name[Category.EXPLORATION].active_tokens == user_query_tokens + expected_tool_tokens + 100
