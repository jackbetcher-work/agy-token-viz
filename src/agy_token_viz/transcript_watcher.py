"""Real-time watcher for Antigravity (`agy`) and Jetski (`jetski`) transcript JSONL files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable, Sequence

from agy_token_viz.config import extract_data_dir_from_args
from agy_token_viz.state import StepRecord, ToolCallSummary
from agy_token_viz.tokenizer import count_tokens_locally, serialize_tool_calls_verbatim


_UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UUID_RE = re.compile(_UUID_PATTERN)
_CLI_LOG_APP_DATA_DIR_RE = re.compile(r"\bCLI app data directory:\s+(.+)$")
_CLI_LOG_CREATED_CONV_RE = re.compile(rf"\bCreated conversation ({_UUID_PATTERN})\b")
_CLI_LOG_ATTACHED_CONV_RE = re.compile(
    rf"\b(?:Streaming conversation|Starting conversation update stream for|"
    rf"Forwarding user message to conversation|Sending user message to conversation|"
    rf"Full redraw completed \(RerenderAll\) for conversation)\s+({_UUID_PATTERN})\b"
)


def _conversation_id_from_path_arg(raw_path: str) -> str | None:
    cleaned = raw_path.strip()
    if not cleaned:
        return None
    uuid_match = _UUID_RE.search(cleaned)
    if uuid_match:
        return uuid_match.group(0)
    stem = Path(cleaned).stem.strip()
    return stem or None


def extract_target_conversation_from_args(agy_args: Sequence[str]) -> str | None:
    """Inspect forwarded CLI flags for an explicit `--conversation` or `--conversation_path` target."""
    conv_flags = {"--conversation", "-conversation"}
    conv_path_flags = {
        "--conversation_path",
        "-conversation_path",
        "--conversation-path",
        "-conversation-path",
    }
    args = list(agy_args)
    for idx, token in enumerate(args):
        if token == "--":
            break
        if token in conv_flags and idx + 1 < len(args):
            val = args[idx + 1].strip()
            return val or None
        for prefix in conv_flags:
            if token.startswith(f"{prefix}="):
                val = token.split("=", 1)[1].strip()
                return val or None
        if token in conv_path_flags and idx + 1 < len(args):
            return _conversation_id_from_path_arg(args[idx + 1])
        for prefix in conv_path_flags:
            if token.startswith(f"{prefix}="):
                return _conversation_id_from_path_arg(token.split("=", 1)[1])
    return None


def is_resuming_conversation_from_args(agy_args: Sequence[str]) -> bool:
    """Return True if forwarded CLI flags resume an existing conversation (`-c`, `--continue`, `--conversation`, `--conversation_path`)."""
    resume_flags = {
        "-c",
        "--continue",
        "-continue",
        "--conversation",
        "-conversation",
        "--conversation_path",
        "-conversation_path",
        "--conversation-path",
        "-conversation-path",
    }
    resume_prefixes = (
        "--conversation=",
        "-conversation=",
        "--conversation_path=",
        "-conversation_path=",
        "--conversation-path=",
        "-conversation-path=",
    )
    for token in agy_args:
        if token == "--":
            break
        if token in resume_flags or token.startswith(resume_prefixes):
            return True
    return False


def extract_log_file_from_args(agy_args: Sequence[str]) -> Path | None:
    """Inspect forwarded CLI flags for an explicit `--log-file` or `--cli_log_file` override."""
    log_flags = {
        "--log-file",
        "-log-file",
        "--cli_log_file",
        "-cli_log_file",
        "--cli-log-file",
        "-cli-log-file",
    }
    args = list(agy_args)
    for idx, token in enumerate(args):
        if token == "--":
            break
        if token in log_flags and idx + 1 < len(args):
            val = args[idx + 1].strip()
            return Path(val).expanduser() if val else None
        for prefix in log_flags:
            if token.startswith(f"{prefix}="):
                val = token.split("=", 1)[1].strip()
                return Path(val).expanduser() if val else None
    return None


def _summarize_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    """Create a short human-readable hint from tool call arguments for classification."""
    if tool_name == "swarm":
        action = str(args.get("action") or "").strip()
        detail = str(
            args.get("name") or args.get("title") or args.get("brief") or ""
        ).strip()
        combined = f"{action} {detail}".strip() if action else detail
        if combined:
            return combined[:240]

    for key in (
        "Path",
        "CommandLine",
        "query",
        "toolSummary",
        "Pattern",
        "Role",
        "Description",
        "ScriptPath",
        "Action",
        "action",
        "brief",
        "Name",
        "Prompt",
    ):
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
        agy_data_dir: Path | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
        target_conversation_id: str | None = None,
        include_history: bool = False,
        poll_interval: float = 0.15,
        *,
        data_dir: Path | None = None,
        log_subdirs: Sequence[str] = ("log", "cli/log"),
        exclude_preexisting: bool = False,
        lock_to_first_conversation: bool = False,
        on_baseline_context: Callable[[str, int, int, int], None] | None = None,
        on_reset: Callable[[], None] | None = None,
        explicit_log_file: Path | None = None,
    ) -> None:
        resolved_dir = data_dir if data_dir is not None else agy_data_dir
        if resolved_dir is None:
            raise TypeError("TranscriptWatcher requires `agy_data_dir` (or `data_dir`).")
        if on_step is None:
            raise TypeError("TranscriptWatcher requires `on_step`.")
        self._agy_data_dir = resolved_dir.expanduser()
        self._brain_dir = self._agy_data_dir / "brain"
        self._log_subdirs: tuple[str, ...] = tuple(log_subdirs) or ("log", "cli/log")
        self._log_dir = self._agy_data_dir / self._log_subdirs[0]
        self._on_step = on_step
        self._target_conversation_id = target_conversation_id
        self._allowed_conversations: set[str] = (
            {target_conversation_id} if target_conversation_id else set()
        )
        self._subagent_conversations: set[str] = set()
        self._preexisting_conversations: set[str] = set()
        self._excluded_conversations: set[str] = set()
        self._rewound_history_conversations: set[str] = set()
        self._active_primary_conversation: str | None = target_conversation_id
        self._include_history = include_history
        self._poll_interval = poll_interval
        self._exclude_preexisting = exclude_preexisting
        self._lock_to_first_conversation = lock_to_first_conversation
        self._on_baseline_context = on_baseline_context
        self._on_reset = on_reset
        self._explicit_log_file = explicit_log_file

        self._child_pid: int | None = None
        self._cli_log_path: Path | None = None
        self._cli_log_offset: int = 0

        self._cursors: dict[Path, _FileCursor] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_captured = False

    def set_child_pid(self, pid: int) -> None:
        """Bind watcher to the spawned `agy` child process PID to track its CLI log."""
        if pid > 0:
            self._child_pid = pid

    def capture_baseline(self) -> None:
        """
        Snapshot existing transcript file byte offsets BEFORE spawning `agy`.

        This guarantees:
        - Fresh `agy` runs (`exclude_preexisting=True`) ignore pre-existing conversations
          (including sessions actively running in other terminals) unless explicitly
          attached to by this `agy` process in its CLI log.
        - `--continue` or resumed conversations only tally NEW steps generated in this run
          (unless `include_history=True`) while seeding pre-session context size so
          uncached prefix-cache tails from prior runs are not re-billed as new input.
        - Newly created conversations start at byte offset 0.
        """
        self._baseline_captured = True
        if not self._brain_dir.is_dir():
            return

        try:
            for child in self._brain_dir.iterdir():
                if child.is_dir():
                    self._preexisting_conversations.add(child.name)
        except OSError:
            pass

        if self._exclude_preexisting and self._target_conversation_id is None:
            self._excluded_conversations.update(self._preexisting_conversations)
            return

        if self._target_conversation_id is not None:
            target_path = self._transcript_path_for(self._target_conversation_id)
            transcript_paths = [target_path] if target_path.is_file() else []
            if target_path.is_file():
                self._preexisting_conversations.add(self._target_conversation_id)
        else:
            transcript_paths = self._iter_transcript_paths()
            for transcript_path in transcript_paths:
                conv_id = self._conversation_id_for(transcript_path)
                self._preexisting_conversations.add(conv_id)

        # When include_history=True without an explicit target_conversation_id, only replay
        # the most recently modified transcript at baseline (not every historical session).
        most_recent_path: Path | None = None
        if (
            self._include_history
            and self._target_conversation_id is None
            and not self._exclude_preexisting
            and transcript_paths
        ):
            try:
                most_recent_path = max(
                    transcript_paths,
                    key=lambda p: p.stat().st_mtime,
                )
            except OSError:
                most_recent_path = transcript_paths[-1]

        for transcript_path in transcript_paths:
            conv_id = self._conversation_id_for(transcript_path)
            replay_from_start = False
            if self._include_history:
                if self._target_conversation_id is not None:
                    replay_from_start = conv_id == self._target_conversation_id
                elif not self._exclude_preexisting and transcript_path == most_recent_path:
                    replay_from_start = True

            if replay_from_start:
                self._rewound_history_conversations.add(conv_id)
                self._cursors[transcript_path] = _FileCursor(
                    conversation_id=conv_id,
                    path=transcript_path,
                    full_path=transcript_path.with_name("transcript_full.jsonl"),
                    byte_offset=0,
                    line_index=0,
                )
            else:
                cursor = self._inspect_existing_transcript(transcript_path, conv_id)
                self._cursors[transcript_path] = cursor

    def _inspect_existing_transcript(self, path: Path, conv_id: str) -> _FileCursor:
        """Measure complete lines at EOF and extract prior context baseline for resumed sessions."""
        full_path = path.with_name("transcript_full.jsonl")
        try:
            data = path.read_bytes()
        except OSError:
            return _FileCursor(
                conversation_id=conv_id,
                path=path,
                full_path=full_path,
                byte_offset=0,
                line_index=0,
            )

        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            return _FileCursor(
                conversation_id=conv_id,
                path=path,
                full_path=full_path,
                byte_offset=0,
                line_index=0,
            )

        consumed = data[: last_nl + 1]
        offset = len(consumed)
        lines = consumed.decode("utf-8", errors="replace").splitlines()
        line_count = len(lines)

        saw_user_input = False
        saw_planner_step = False
        saw_checkpoint = False
        last_user_input = ""
        pending_payload_parts: list[str] = []
        last_planner_obj: dict[str, Any] | None = None
        last_planner_line_idx: int = -1

        # Scan backwards from EOF to locate the most recent completed PLANNER_RESPONSE
        # without JSON-parsing earlier historical turns that would be overwritten.
        for idx in range(line_count - 1, -1, -1):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "PLANNER_RESPONSE" and obj.get("status") in (None, "DONE"):
                saw_planner_step = True
                last_planner_obj = obj
                last_planner_line_idx = idx
                break

        # Process only the trailing events after the last completed PLANNER_RESPONSE
        for idx in range(last_planner_line_idx + 1, line_count):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            entry_type = obj.get("type")
            if entry_type == "CHECKPOINT":
                saw_checkpoint = True
                pending_payload_parts.clear()
            elif entry_type == "USER_INPUT":
                saw_user_input = True
                text = str(obj.get("content") or "")
                last_user_input = text
                if text:
                    pending_payload_parts.append(text)
            elif entry_type in ("GENERIC", "SYSTEM_MESSAGE", "ERROR_MESSAGE"):
                if obj.get("status") in (None, "DONE", "ERROR"):
                    text = str(obj.get("content") or "")
                    if text:
                        pending_payload_parts.append(text)

        if saw_planner_step and last_planner_obj is not None and self._on_baseline_context is not None:
            if last_planner_obj.get("truncated_fields") and last_planner_line_idx >= 0:
                full_line = _read_specific_line(full_path, last_planner_line_idx)
                if full_line:
                    try:
                        hydrated = json.loads(full_line)
                        if isinstance(hydrated, dict):
                            last_planner_obj = hydrated
                    except json.JSONDecodeError:
                        pass
            last_step = parse_planner_step(last_planner_obj, conv_id)
            if last_step is not None:
                prompt_tok = max(0, last_step.cache_read_tokens) + max(0, last_step.input_tokens)
                vis_end = prompt_tok + last_step.visible_output_tokens
                end_ctx = prompt_tok + max(0, last_step.output_tokens)
                self._on_baseline_context(conv_id, prompt_tok, vis_end, end_ctx)

        return _FileCursor(
            conversation_id=conv_id,
            path=path,
            full_path=full_path,
            byte_offset=offset,
            line_index=line_count,
            saw_user_input=saw_user_input,
            saw_planner_step=saw_planner_step,
            saw_checkpoint=saw_checkpoint,
            last_user_input=last_user_input,
            pending_payload_parts=pending_payload_parts,
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

    def _discover_cli_log_for_pid(self, pid: int) -> Path | None:
        """Locate the `cli-*.log` file created by our spawned `agy` or `jetski` child process `pid`."""
        if self._explicit_log_file is not None and self._explicit_log_file.is_file():
            return self._explicit_log_file

        log_dirs = [
            d
            for sub in self._log_subdirs
            if (d := self._agy_data_dir / sub).is_dir()
        ]
        if not log_dirs:
            return None

        pid_pattern = re.compile(
            rf"\b(?:Starting language server process with pid|Process id)\s+{pid}\b"
        )
        seen_paths: set[Path] = set()
        weighted_candidates: list[tuple[float, Path]] = []
        for log_dir in log_dirs:
            try:
                for path in log_dir.glob("cli-*.log"):
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    if st.st_size <= 0:
                        continue
                    weighted_candidates.append((st.st_mtime, path))
            except OSError:
                continue

        if not weighted_candidates:
            return None

        weighted_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = [path for _, path in weighted_candidates[:16]]

        for cand in candidates:
            try:
                with cand.open("rb") as fh:
                    head = fh.read(8192).decode("utf-8", errors="replace")
            except OSError:
                continue
            if pid_pattern.search(head):
                return cand
        return None

    def _sync_cli_log(self) -> None:
        """Tail the child CLI process's log to bind strictly to its active conversation(s)."""
        if self._cli_log_path is None and self._child_pid is not None:
            found = self._discover_cli_log_for_pid(self._child_pid)
            if found is not None:
                self._cli_log_path = found
                self._cli_log_offset = 0

        if self._cli_log_path is None:
            return

        try:
            with self._cli_log_path.open("rb") as fh:
                fh.seek(self._cli_log_offset)
                chunk = fh.read()
        except OSError:
            return

        if not chunk:
            return

        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            return

        complete = chunk[: last_nl + 1]
        self._cli_log_offset += len(complete)

        for line in complete.decode("utf-8", errors="replace").splitlines():
            app_dir_match = _CLI_LOG_APP_DATA_DIR_RE.search(line)
            if app_dir_match:
                reported_dir = Path(app_dir_match.group(1).strip()).expanduser()
                try:
                    if (
                        reported_dir.is_dir()
                        and reported_dir.resolve() != self._agy_data_dir.resolve()
                    ):
                        self._agy_data_dir = reported_dir
                        self._brain_dir = reported_dir / "brain"
                except OSError:
                    pass
                continue

            created_match = _CLI_LOG_CREATED_CONV_RE.search(line)
            if created_match:
                conv_id = created_match.group(1)
                if (
                    self._active_primary_conversation is not None
                    and conv_id != self._active_primary_conversation
                    and conv_id not in self._subagent_conversations
                ):
                    # User started a brand-new conversation interactively (e.g., `/new`)
                    if self._on_reset is not None:
                        self._on_reset()
                    self._allowed_conversations = {conv_id}
                    self._subagent_conversations.clear()
                else:
                    self._allowed_conversations.add(conv_id)
                self._active_primary_conversation = conv_id
                self._excluded_conversations.discard(conv_id)
                continue

            attached_match = _CLI_LOG_ATTACHED_CONV_RE.search(line)
            if attached_match:
                conv_id = attached_match.group(1)
                if self._active_primary_conversation is None:
                    self._active_primary_conversation = conv_id
                    self._allowed_conversations.add(conv_id)
                elif (
                    conv_id != self._active_primary_conversation
                    and conv_id not in self._allowed_conversations
                    and conv_id not in self._subagent_conversations
                ):
                    # User switched to / resumed a different conversation interactively
                    if self._on_reset is not None:
                        self._on_reset()
                    self._allowed_conversations = {conv_id}
                    self._subagent_conversations.clear()
                    self._active_primary_conversation = conv_id
                else:
                    self._allowed_conversations.add(conv_id)
                self._excluded_conversations.discard(conv_id)

                transcript_path = self._transcript_path_for(conv_id)
                if (
                    conv_id in self._preexisting_conversations
                    and not self._include_history
                    and transcript_path not in self._cursors
                    and transcript_path.is_file()
                ):
                    self._cursors[transcript_path] = self._inspect_existing_transcript(
                        transcript_path, conv_id
                    )

                if self._include_history and conv_id not in self._rewound_history_conversations:
                    self._rewound_history_conversations.add(conv_id)
                    for cur in self._cursors.values():
                        if cur.conversation_id == conv_id and cur.byte_offset > 0:
                            cur.byte_offset = 0
                            cur.line_index = 0
                            cur.saw_user_input = False
                            cur.saw_planner_step = False
                            cur.saw_checkpoint = False
                            cur.last_user_input = ""
                            cur.pending_payload_parts.clear()

    def poll_once(self) -> int:
        """Scan for new complete JSONL lines across tracked/newly created transcripts."""
        self._sync_cli_log()

        if not self._brain_dir.is_dir():
            return 0

        if self._cli_log_path is not None or self._allowed_conversations:
            # Fast path: check only bound/allowed conversations without globbing brain/*
            transcript_paths = [
                p
                for conv_id in sorted(self._allowed_conversations)
                if (p := self._transcript_path_for(conv_id)).is_file()
            ]
        else:
            transcript_paths = self._iter_transcript_paths()

        emitted = 0
        for transcript_path in transcript_paths:
            conv_id = self._conversation_id_for(transcript_path)

            if self._cli_log_path is not None:
                # Strictly bound to conversations owned by our spawned `agy` process
                if conv_id not in self._allowed_conversations:
                    continue
            else:
                if conv_id in self._excluded_conversations:
                    continue
                if self._allowed_conversations and conv_id not in self._allowed_conversations:
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

            # Track spawned subagent conversation IDs for the active session
            self._discover_subagents(obj, cursor.conversation_id)

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
                if self._lock_to_first_conversation and not self._allowed_conversations:
                    self._active_primary_conversation = cursor.conversation_id
                    self._allowed_conversations.add(cursor.conversation_id)
                self._on_step(step)
                emitted += 1

        return emitted

    def _discover_subagents(self, obj: dict[str, Any], parent_conv_id: str) -> None:
        if (
            self._target_conversation_id is None
            and not self._lock_to_first_conversation
            and not self._allowed_conversations
        ):
            return

        found_ids: list[str] = []
        content = obj.get("content")
        if isinstance(content, str) and (
            "conversationId" in content
            or "conversation_id" in content
            or "conversation://" in content
        ):
            found_ids.extend(_UUID_RE.findall(content))

        sub_info = obj.get("subagent_info")
        if isinstance(sub_info, (dict, list)):
            try:
                found_ids.extend(_UUID_RE.findall(json.dumps(sub_info)))
            except (TypeError, ValueError):
                pass

        for match in found_ids:
            if match == parent_conv_id or match == self._active_primary_conversation:
                continue
            if self._lock_to_first_conversation and not self._allowed_conversations:
                self._active_primary_conversation = parent_conv_id
                self._allowed_conversations.add(parent_conv_id)
            self._subagent_conversations.add(match)
            self._allowed_conversations.add(match)
            self._excluded_conversations.discard(match)

    def _transcript_path_for(self, conv_id: str) -> Path:
        return (
            self._brain_dir
            / conv_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )

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
