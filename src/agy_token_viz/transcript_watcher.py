"""Real-time watcher for Antigravity CLI transcript.jsonl and transcript_full.jsonl files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Callable, Sequence

from agy_token_viz.state import StepRecord, ToolCallSummary
from agy_token_viz.tokenizer import count_tokens_locally, serialize_tool_calls_verbatim


def extract_target_conversation_from_args(agy_args: Sequence[str]) -> str | None:
    """Inspect forwarded `agy` CLI flags for an explicit `--conversation <id>` argument."""
    args = list(agy_args)
    for idx, token in enumerate(args):
        if token == "--conversation" and idx + 1 < len(args):
            val = args[idx + 1].strip()
            return val or None
        if token.startswith("--conversation="):
            val = token.split("=", 1)[1].strip()
            return val or None
    return None


def _summarize_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    """Create a short human-readable hint from tool call arguments for classification."""
    for key in ("Path", "CommandLine", "query", "toolSummary", "Pattern", "Role"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            cleaned = val.strip().strip('"').strip("'")
            return cleaned[:240]
    return ""


def _estimate_visible_output_tokens(content: str, raw_tools: Any) -> int:
    """
    Compute visible assistant output tokens from verbatim `content` + serialized `tool_calls`
    using the local Gemini/Gemma SentencePiece tokenizer.
    """
    content_text = content or ""
    serialized_tools = serialize_tool_calls_verbatim(raw_tools)
    visible_payload = content_text + serialized_tools
    if not visible_payload:
        return 0
    return max(1, count_tokens_locally(visible_payload))


def _resolve_thinking_tokens(
    payload: dict[str, Any],
    out_tokens: int,
    content: str,
    raw_tools: Any,
    thinking: str,
) -> int:
    """Extract explicit thinking tokens if present, or compute exact residual thinking tokens."""
    if out_tokens <= 0:
        return 0

    for key in ("thinking_tokens", "thoughts_token_count", "reasoning_tokens"):
        if key in payload and payload[key] is not None:
            try:
                return max(0, min(out_tokens, int(payload[key])))
            except (TypeError, ValueError):
                pass

    est_vis = _estimate_visible_output_tokens(content, raw_tools)
    if thinking.strip():
        vis_tok = min(out_tokens, est_vis)
        return max(0, out_tokens - vis_tok)

    # Google models via `agy` often omit `thinking` text on tool-call turns while still billing
    # `thoughtsTokenCount` inside `output_tokens`. Allow a small structural framing margin when
    # `thinking` is empty so minor function-call wrapper tokens do not trigger false positives.
    margin = max(16, int(est_vis * 0.08))
    if out_tokens > est_vis and (out_tokens - est_vis) >= margin:
        return out_tokens - est_vis
    return 0


def _extract_initial_user_message(
    payload: dict[str, Any],
    fallback_user_input: str = "",
) -> str:
    """Locate the initial user query from step payload fields or the preceding USER_INPUT event."""
    for key in ("query", "user_prompt", "user_initial_message"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
    return fallback_user_input


def parse_planner_step(
    payload: dict[str, Any],
    conversation_id: str,
    *,
    preceded_by_user_input: bool = False,
    is_first_step: bool = False,
    initial_user_input: str = "",
    appended_payload: str = "",
    compacted: bool = False,
) -> StepRecord | None:
    """Parse a JSONL line into a StepRecord if it is a completed PLANNER_RESPONSE."""
    if payload.get("type") != "PLANNER_RESPONSE":
        return None
    status = payload.get("status")
    if status and status != "DONE":
        return None

    step_idx_raw = payload.get("step_index", 0)
    try:
        step_index = int(step_idx_raw)
    except (TypeError, ValueError):
        return None

    def _safe_int(key: str) -> int:
        val = payload.get(key, 0)
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            return 0

    in_tokens = _safe_int("input_tokens")
    cache_tokens = _safe_int("cache_read_tokens")
    out_tokens = _safe_int("output_tokens")

    thinking = str(payload.get("thinking") or "")
    content = str(payload.get("content") or "")

    tool_summaries: list[ToolCallSummary] = []
    raw_tools = payload.get("tool_calls")
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            raw_args = item.get("args")
            args_dict = raw_args if isinstance(raw_args, dict) else {}
            tool_summaries.append(
                ToolCallSummary(
                    name=name,
                    summary=_summarize_tool_args(name, args_dict),
                )
            )

    thinking_tokens = _resolve_thinking_tokens(
        payload=payload,
        out_tokens=out_tokens,
        content=content,
        raw_tools=raw_tools,
        thinking=thinking,
    )

    user_initial_message = _extract_initial_user_message(payload, initial_user_input)
    harness_preamble_tokens = 0
    for key in ("harness_overhead_tokens", "harness_preamble_tokens"):
        if key in payload and payload[key] is not None:
            try:
                harness_preamble_tokens = max(0, min(in_tokens, int(payload[key])))
                break
            except (TypeError, ValueError):
                pass
    else:
        if cache_tokens == 0 and (is_first_step or step_index <= 1) and user_initial_message:
            user_prompt_tokens = count_tokens_locally(user_initial_message)
            harness_preamble_tokens = max(0, in_tokens - user_prompt_tokens)

    resolved_appended = appended_payload
    if not resolved_appended:
        for key in ("appended_payload", "tool_result"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                resolved_appended = val
                break

    return StepRecord(
        conversation_id=conversation_id,
        step_index=step_index,
        input_tokens=in_tokens,
        cache_read_tokens=cache_tokens,
        output_tokens=out_tokens,
        thinking=thinking,
        content=content,
        tool_calls=tuple(tool_summaries),
        created_at=str(payload.get("created_at") or ""),
        preceded_by_user_input=preceded_by_user_input,
        thinking_tokens=thinking_tokens,
        harness_overhead_tokens=harness_preamble_tokens,
        user_prompt=user_initial_message if (is_first_step or step_index <= 1) else "",
        appended_payload=resolved_appended,
        compacted=compacted or bool(payload.get("compacted")),
    )


def _read_specific_line(path: Path, target_line_zero_based: int) -> str | None:
    """Read only a single 0-based line from `transcript_full.jsonl` when truncation occurred."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                if idx == target_line_zero_based:
                    return line
                if idx > target_line_zero_based:
                    break
    except OSError:
        return None
    return None


@dataclass
class _FileCursor:
    conversation_id: str
    path: Path
    full_path: Path
    byte_offset: int
    line_index: int
    saw_user_input: bool = False
    saw_planner_step: bool = False
    saw_checkpoint: bool = False
    last_user_input: str = ""
    pending_payload_parts: list[str] = field(default_factory=list)


class TranscriptWatcher:
    """
    Polls Antigravity CLI's `brain/<conversation-id>/.system_generated/logs/transcript.jsonl`
    for newly appended `PLANNER_RESPONSE` steps during the active CLI session.
    """

    def __init__(
        self,
        agy_data_dir: Path,
        on_step: Callable[[StepRecord], None],
        target_conversation_id: str | None = None,
        include_history: bool = False,
        poll_interval: float = 0.15,
    ) -> None:
        self._brain_dir = agy_data_dir.expanduser() / "brain"
        self._on_step = on_step
        self._target_conversation_id = target_conversation_id
        self._allowed_conversations: set[str] = (
            {target_conversation_id} if target_conversation_id else set()
        )
        self._include_history = include_history
        self._poll_interval = poll_interval

        self._cursors: dict[Path, _FileCursor] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_captured = False

    def capture_baseline(self) -> None:
        """
        Snapshot existing transcript file byte offsets BEFORE spawning `agy`.

        This guarantees:
        - `--continue` or resumed conversations only tally NEW steps generated in this run
          (unless `include_history=True`).
        - Newly created conversations start at byte offset 0.
        """
        self._baseline_captured = True
        if not self._brain_dir.is_dir():
            return

        for transcript_path in self._iter_transcript_paths():
            conv_id = self._conversation_id_for(transcript_path)
            if self._include_history and (
                self._target_conversation_id is None
                or conv_id == self._target_conversation_id
            ):
                offset = 0
                line_count = 0
            else:
                offset, line_count = self._measure_complete_lines(transcript_path)

            self._cursors[transcript_path] = _FileCursor(
                conversation_id=conv_id,
                path=transcript_path,
                full_path=transcript_path.with_name("transcript_full.jsonl"),
                byte_offset=offset,
                line_index=line_count,
            )

    def start(self) -> None:
        if not self._baseline_captured:
            self.capture_baseline()
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="agy-token-viz-transcript-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # Perform one final poll to catch steps flushed right as `agy` exited
        self.poll_once()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self._poll_interval)

    def poll_once(self) -> int:
        """Scan for new complete JSONL lines across tracked/newly created transcripts."""
        if not self._brain_dir.is_dir():
            return 0

        emitted = 0
        for transcript_path in self._iter_transcript_paths():
            conv_id = self._conversation_id_for(transcript_path)
            if (
                self._target_conversation_id is not None
                and conv_id not in self._allowed_conversations
            ):
                continue

            cursor = self._cursors.get(transcript_path)
            if cursor is None:
                cursor = _FileCursor(
                    conversation_id=conv_id,
                    path=transcript_path,
                    full_path=transcript_path.with_name("transcript_full.jsonl"),
                    byte_offset=0,
                    line_index=0,
                )
                self._cursors[transcript_path] = cursor

            emitted += self._drain_cursor(cursor)
        return emitted

    def _drain_cursor(self, cursor: _FileCursor) -> int:
        try:
            stat = cursor.path.stat()
        except OSError:
            return 0

        if stat.st_size < cursor.byte_offset:
            # File was truncated/recreated
            cursor.byte_offset = 0
            cursor.line_index = 0
            cursor.saw_user_input = False
            cursor.saw_planner_step = False
            cursor.saw_checkpoint = False
            cursor.last_user_input = ""
            cursor.pending_payload_parts.clear()
        elif stat.st_size == cursor.byte_offset:
            return 0

        emitted = 0
        try:
            with cursor.path.open("rb") as fh:
                fh.seek(cursor.byte_offset)
                raw_chunk = fh.read()
        except OSError:
            return 0

        if not raw_chunk:
            return 0

        # Only consume up to the last newline so partial writes wait for next poll
        last_nl = raw_chunk.rfind(b"\n")
        if last_nl == -1:
            return 0

        complete_bytes = raw_chunk[: last_nl + 1]
        cursor.byte_offset += len(complete_bytes)

        lines = complete_bytes.decode("utf-8", errors="replace").splitlines()
        for raw_line in lines:
            current_line_idx = cursor.line_index
            cursor.line_index += 1
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            # Hydrate from transcript_full.jsonl if any fields were truncated
            if obj.get("truncated_fields"):
                full_line = _read_specific_line(cursor.full_path, current_line_idx)
                if full_line:
                    try:
                        full_obj = json.loads(full_line)
                        if isinstance(full_obj, dict):
                            obj = full_obj
                    except json.JSONDecodeError:
                        pass

            # Track spawned subagent conversation IDs when targeting a specific parent session
            self._discover_subagents(obj)

            entry_type = obj.get("type")
            if entry_type == "CHECKPOINT":
                cursor.saw_checkpoint = True
                cursor.pending_payload_parts.clear()
                continue

            if entry_type == "USER_INPUT":
                cursor.saw_user_input = True
                text = str(obj.get("content") or "")
                cursor.last_user_input = text
                if text:
                    cursor.pending_payload_parts.append(text)
                continue

            if entry_type in ("GENERIC", "SYSTEM_MESSAGE", "ERROR_MESSAGE"):
                if obj.get("status") in (None, "DONE", "ERROR"):
                    text = str(obj.get("content") or "")
                    if text:
                        cursor.pending_payload_parts.append(text)
                continue

            step = parse_planner_step(
                obj,
                cursor.conversation_id,
                preceded_by_user_input=cursor.saw_user_input,
                is_first_step=not cursor.saw_planner_step,
                initial_user_input=cursor.last_user_input if not cursor.saw_planner_step else "",
                appended_payload="\n".join(cursor.pending_payload_parts),
                compacted=cursor.saw_checkpoint,
            )
            if step is not None:
                cursor.saw_user_input = False
                cursor.saw_planner_step = True
                cursor.saw_checkpoint = False
                cursor.pending_payload_parts.clear()
                self._on_step(step)
                emitted += 1

        return emitted

    def _discover_subagents(self, obj: dict[str, Any]) -> None:
        if self._target_conversation_id is None:
            return
        content = obj.get("content")
        if isinstance(content, str) and "conversationId" in content:
            # Extract UUID-like conversation IDs returned by invoke_subagent
            import re

            for match in re.findall(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                content,
            ):
                self._allowed_conversations.add(match)

    def _iter_transcript_paths(self) -> list[Path]:
        try:
            return sorted(
                self._brain_dir.glob("*/.system_generated/logs/transcript.jsonl")
            )
        except OSError:
            return []

    @staticmethod
    def _conversation_id_for(transcript_path: Path) -> str:
        # .../<conversation-id>/.system_generated/logs/transcript.jsonl
        return transcript_path.parents[2].name

    @staticmethod
    def _measure_complete_lines(path: Path) -> tuple[int, int]:
        try:
            data = path.read_bytes()
        except OSError:
            return 0, 0
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            return 0, 0
        consumed = data[: last_nl + 1]
        return len(consumed), consumed.count(b"\n")
