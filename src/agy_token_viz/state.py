"""In-memory thread-safe token accounting state by activity category."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from agy_token_viz.config import DEFAULT_CATEGORIES, HARNESS_OVERHEAD_CATEGORY
from agy_token_viz.tokenizer import count_tokens_locally


class Category:
    """Canonical activity and system overhead category constants."""

    PLANNING = "Planning"
    EXPLORATION = "Exploration"
    CODING = "Coding"
    COMMANDS = "Commands"
    TESTING = "Testing"
    COMMUNICATION = "Communication"
    HARNESS_OVERHEAD = HARNESS_OVERHEAD_CATEGORY
    SYSTEM_OVERHEAD = "System Overhead"
    SYSTEM_SETUP = "System Setup"


@dataclass(frozen=True)
class ToolCallSummary:
    """Lightweight representation of a tool call made during an agent step."""

    name: str
    summary: str = ""


@dataclass(frozen=True)
class StepRecord:
    """Parsed PLANNER_RESPONSE step from Antigravity's transcript.jsonl."""

    conversation_id: str
    step_index: int
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    thinking: str = ""
    content: str = ""
    tool_calls: tuple[ToolCallSummary, ...] = ()
    created_at: str = ""
    preceded_by_user_input: bool = False
    thinking_tokens: int = 0
    harness_overhead_tokens: int = 0
    user_prompt: str = ""
    appended_payload: str = ""
    compacted: bool = False

    @property
    def visible_output_tokens(self) -> int:
        """Estimated or reported non-thinking output tokens (visible reply + tool calls)."""
        out_tok = max(0, self.output_tokens)
        think_tok = max(0, min(out_tok, self.thinking_tokens))
        return max(0, out_tok - think_tok)

    @property
    def active_tokens(self) -> int:
        """Raw step Input + Output tokens as reported on the transcript record."""
        return max(0, self.input_tokens) + max(0, self.output_tokens)


def extract_new_step_payload(step: StepRecord | Mapping[str, Any]) -> str:
    """Extract newly appended event text (tool results, user prompt, or system message) for a step."""
    if isinstance(step, StepRecord):
        if step.appended_payload:
            return step.appended_payload
        if step.user_prompt:
            return step.user_prompt
        return ""
    if isinstance(step, Mapping):
        for key in ("appended_payload", "tool_result", "user_prompt", "query"):
            val = step.get(key)
            if isinstance(val, str) and val:
                return val
    return ""


@dataclass(frozen=True)
class CategorySnapshot:
    """Immutable snapshot of a single category's cumulative usage."""

    name: str
    active_tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    steps: int
    percentage: float
    thinking_tokens: int = 0


@dataclass(frozen=True)
class LedgerSnapshot:
    """Immutable snapshot of the entire in-memory session token ledger."""

    categories: tuple[CategorySnapshot, ...]
    total_active_tokens: int
    classified_active_tokens: int
    unclassified_active_tokens: int
    total_input_tokens: int
    total_billed_input_tokens: int
    total_cache_read_tokens: int
    total_output_tokens: int
    total_steps: int
    classified_steps: int
    pending_steps: int
    router_status: str
    last_category: str | None
    elapsed_seconds: float
    total_thinking_tokens: int = 0
    total_harness_overhead_tokens: int = 0


@dataclass
class _MutableCategoryTotals:
    active_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    steps: int = 0


@dataclass
class _PendingStep:
    record: StepRecord
    net_input_tokens: int
    output_tokens: int
    thinking_tokens: int
    visible_output_tokens: int
    cache_read_tokens: int


@dataclass
class _ConversationContextState:
    prev_prompt_tokens: int = 0
    prev_visible_end_context: int = 0
    prev_end_context: int = 0
    last_tool_step_key: tuple[str, int] | None = None


class SessionTokenState:
    """
    Thread-safe in-memory ledger tracking agent token usage across categories.

    Uses monotonic net context deltas per conversation (`(cache + in) - prev_end_context`)
    so prefix-cache block-tail re-reads and prior assistant outputs are not double-counted,
    isolates Step 1 harness preamble overhead (`Harness Overhead`), and falls back to
    tokenizing newly appended payloads when thought blocks are pruned across turn boundaries.
    """

    def __init__(
        self,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._categories: tuple[str, ...] = tuple(categories) if categories else DEFAULT_CATEGORIES
        self._category_lookup: dict[str, str] = {c.lower(): c for c in self._categories}
        self._buckets: dict[str, _MutableCategoryTotals] = {
            c: _MutableCategoryTotals() for c in self._categories
        }
        self._seen_steps: set[tuple[str, int]] = set()
        self._pending_records: dict[tuple[str, int], _PendingStep] = {}
        self._classified_step_category: dict[tuple[str, int], str] = {}
        self._conversations: dict[str, _ConversationContextState] = {}

        self._total_input_tokens: int = 0
        self._total_billed_input_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_thinking_tokens: int = 0
        self._total_harness_overhead_tokens: int = 0
        self._unclassified_active_tokens: int = 0
        self._classified_steps: int = 0
        self._router_status: str = "warming up"
        self._last_category: str | None = None
        self._started_at: float = time.monotonic()
        self._on_change = on_change

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    def set_on_change(self, callback: Callable[[], None] | None) -> None:
        with self._lock:
            self._on_change = callback

    def set_router_status(self, status: str) -> None:
        cb: Callable[[], None] | None = None
        with self._lock:
            if self._router_status != status:
                self._router_status = status
                cb = self._on_change
        if cb is not None:
            cb()

    def register_step(self, record: StepRecord) -> bool:
        """Record a newly observed step immediately; classification completes asynchronously."""
        key = (record.conversation_id, record.step_index)
        cb: Callable[[], None] | None = None
        with self._lock:
            if key in self._seen_steps:
                return False
            self._seen_steps.add(key)

            raw_in = max(0, record.input_tokens)
            cache_tok = max(0, record.cache_read_tokens)
            out_tok = max(0, record.output_tokens)
            think_tok = max(0, min(out_tok, record.thinking_tokens))
            vis_out_tok = max(0, out_tok - think_tok)
            prompt_tokens = cache_tok + raw_in

            conv = self._conversations.get(record.conversation_id)
            attributed_prev_key: tuple[str, int] | None = None
            harness_tok = 0
            if conv is None:
                net_in = raw_in
                if cache_tok == 0:
                    if record.harness_overhead_tokens > 0:
                        harness_tok = min(net_in, record.harness_overhead_tokens)
                    elif record.user_prompt:
                        user_prompt_tokens = count_tokens_locally(record.user_prompt)
                        harness_tok = max(0, net_in - user_prompt_tokens)
                conv = _ConversationContextState(
                    prev_prompt_tokens=prompt_tokens,
                    prev_visible_end_context=prompt_tokens + vis_out_tok,
                    prev_end_context=prompt_tokens + out_tok,
                    last_tool_step_key=key if record.tool_calls else None,
                )
                self._conversations[record.conversation_id] = conv
            else:
                if record.compacted:
                    # Explicit context compaction (/compact) event; synchronize baseline without re-billing
                    net_in = 0
                elif prompt_tokens < conv.prev_end_context:
                    # Prior thought blocks were stripped across turn/error boundaries
                    appended_payload = extract_new_step_payload(record)
                    if appended_payload:
                        net_in = count_tokens_locally(appended_payload)
                    elif (
                        prompt_tokens >= conv.prev_visible_end_context
                        and conv.prev_end_context > conv.prev_visible_end_context
                    ):
                        net_in = max(0, prompt_tokens - conv.prev_visible_end_context)
                    else:
                        # Unmarked context compaction where prompt shrunk and no appended payload was provided
                        net_in = 0
                else:
                    # Standard monotonic progression
                    net_in = prompt_tokens - conv.prev_end_context

                if conv.last_tool_step_key is not None and not record.preceded_by_user_input:
                    attributed_prev_key = conv.last_tool_step_key

                conv.prev_prompt_tokens = prompt_tokens
                conv.prev_visible_end_context = prompt_tokens + vis_out_tok
                conv.prev_end_context = prompt_tokens + out_tok
                conv.last_tool_step_key = key if record.tool_calls else None

            self._total_billed_input_tokens += raw_in
            self._total_input_tokens += net_in
            self._total_cache_read_tokens += cache_tok
            self._total_output_tokens += out_tok
            self._total_thinking_tokens += think_tok

            own_net_in = net_in
            if harness_tok > 0:
                harness_cat = self._ensure_harness_category_locked()
                harness_bucket = self._buckets[harness_cat]
                harness_bucket.input_tokens += harness_tok
                harness_bucket.active_tokens += harness_tok
                if harness_bucket.steps == 0:
                    harness_bucket.steps = 1
                self._total_harness_overhead_tokens += harness_tok
                own_net_in = max(0, net_in - harness_tok)

            if attributed_prev_key is not None and own_net_in > 0:
                if attributed_prev_key in self._pending_records:
                    self._pending_records[attributed_prev_key].net_input_tokens += own_net_in
                    self._unclassified_active_tokens += own_net_in
                    own_net_in = 0
                elif attributed_prev_key in self._classified_step_category:
                    prev_cat = self._classified_step_category[attributed_prev_key]
                    prev_bucket = self._buckets[prev_cat]
                    prev_bucket.input_tokens += own_net_in
                    prev_bucket.active_tokens += own_net_in
                    own_net_in = 0

            self._pending_records[key] = _PendingStep(
                record=record,
                net_input_tokens=own_net_in,
                output_tokens=out_tok,
                thinking_tokens=think_tok,
                visible_output_tokens=vis_out_tok,
                cache_read_tokens=cache_tok,
            )
            self._unclassified_active_tokens += own_net_in + out_tok
            cb = self._on_change

        if cb is not None:
            cb()
        return True

    def complete_classification(
        self,
        conversation_id: str,
        step_index: int,
        category: str,
    ) -> str:
        """Assign a previously registered step's tokens to its classified activity category."""
        key = (conversation_id, step_index)
        cb: Callable[[], None] | None = None
        with self._lock:
            canonical = self._resolve_category_locked(category)
            pending = self._pending_records.pop(key, None)
            if pending is None:
                return canonical

            self._classified_step_category[key] = canonical
            in_tok = pending.net_input_tokens
            cache_tok = pending.cache_read_tokens
            out_tok = pending.output_tokens
            think_tok = pending.thinking_tokens
            vis_out_tok = pending.visible_output_tokens
            active = in_tok + out_tok

            self._unclassified_active_tokens = max(0, self._unclassified_active_tokens - active)
            plan_cat = self._find_planning_category_locked()
            if think_tok > 0 and plan_cat is not None and plan_cat != canonical:
                plan_bucket = self._buckets[plan_cat]
                plan_bucket.active_tokens += think_tok
                plan_bucket.output_tokens += think_tok
                plan_bucket.thinking_tokens += think_tok

                bucket = self._buckets[canonical]
                bucket.active_tokens += in_tok + vis_out_tok
                bucket.input_tokens += in_tok
                bucket.output_tokens += vis_out_tok
                bucket.cache_read_tokens += cache_tok
                bucket.steps += 1
            else:
                bucket = self._buckets[canonical]
                bucket.active_tokens += active
                bucket.input_tokens += in_tok
                bucket.output_tokens += out_tok
                bucket.thinking_tokens += think_tok
                bucket.cache_read_tokens += cache_tok
                bucket.steps += 1

            self._classified_steps += 1
            self._last_category = canonical
            cb = self._on_change

        if cb is not None:
            cb()
        return canonical

    def record_classified_step(self, record: StepRecord, category: str) -> bool:
        """Convenience helper to register and immediately classify a step."""
        if not self.register_step(record):
            return False
        self.complete_classification(record.conversation_id, record.step_index, category)
        return True

    def _ensure_harness_category_locked(self) -> str:
        for kw in ("harness overhead", "system overhead", "system setup", "harness", "overhead"):
            for key, orig in self._category_lookup.items():
                if kw in key:
                    return orig
        cat = Category.HARNESS_OVERHEAD
        self._categories = (*self._categories, cat)
        self._category_lookup[cat.lower()] = cat
        self._buckets[cat] = _MutableCategoryTotals()
        return cat

    def _find_planning_category_locked(self) -> str | None:
        if "planning" in self._category_lookup:
            return self._category_lookup["planning"]
        for kw in ("plan", "think", "reason", "design", "arch"):
            for key, orig in self._category_lookup.items():
                if kw in key:
                    return orig
        return None

    def _resolve_category_locked(self, raw_category: str) -> str:
        cleaned = (raw_category or "").strip()
        if cleaned in self._buckets:
            return cleaned
        lowered = cleaned.lower()
        if lowered in self._category_lookup:
            return self._category_lookup[lowered]
        # Prefix / substring match fallback
        for cat in self._categories:
            if cat.lower().startswith(lowered) and lowered:
                return cat
            if lowered and cat.lower() in lowered:
                return cat
        return self._categories[0]

    def snapshot(self) -> LedgerSnapshot:
        """Return an immutable snapshot suitable for rendering."""
        with self._lock:
            total_active = self._total_input_tokens + self._total_output_tokens
            classified_active = sum(b.active_tokens for b in self._buckets.values())
            denom = classified_active if classified_active > 0 else 0

            cat_snaps = tuple(
                CategorySnapshot(
                    name=cat,
                    active_tokens=self._buckets[cat].active_tokens,
                    input_tokens=self._buckets[cat].input_tokens,
                    output_tokens=self._buckets[cat].output_tokens,
                    cache_read_tokens=self._buckets[cat].cache_read_tokens,
                    steps=self._buckets[cat].steps,
                    percentage=(
                        (self._buckets[cat].active_tokens / denom) * 100.0
                        if denom > 0
                        else 0.0
                    ),
                    thinking_tokens=self._buckets[cat].thinking_tokens,
                )
                for cat in self._categories
            )

            return LedgerSnapshot(
                categories=cat_snaps,
                total_active_tokens=total_active,
                classified_active_tokens=classified_active,
                unclassified_active_tokens=self._unclassified_active_tokens,
                total_input_tokens=self._total_input_tokens,
                total_billed_input_tokens=self._total_billed_input_tokens,
                total_cache_read_tokens=self._total_cache_read_tokens,
                total_output_tokens=self._total_output_tokens,
                total_steps=len(self._seen_steps),
                classified_steps=self._classified_steps,
                pending_steps=len(self._pending_records),
                router_status=self._router_status,
                last_category=self._last_category,
                elapsed_seconds=max(0.0, time.monotonic() - self._started_at),
                total_thinking_tokens=self._total_thinking_tokens,
                total_harness_overhead_tokens=self._total_harness_overhead_tokens,
            )
