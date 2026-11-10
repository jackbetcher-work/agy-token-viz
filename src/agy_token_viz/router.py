"""Open Model Router: local Hugging Face GGUF weight management & Metal-accelerated classification."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path, PurePath
import queue
import re
import shlex
import shutil
import threading
from typing import Any, Callable, Protocol, Sequence

from agy_token_viz.config import (
    DEFAULT_CATEGORIES,
    ModelSpec,
    VizConfig,
    get_hf_token,
)
from agy_token_viz.state import SessionTokenState, StepRecord, ToolCallSummary


logger = logging.getLogger(__name__)

# Keep a module-level reference to the ctypes callback so GC never collects it.
_LLAMA_NOOP_LOG_CB: Any = None


def _silence_llama_cpp_logs() -> None:
    """Prevent C-level llama.cpp / Metal init logs from writing to terminal stderr."""
    global _LLAMA_NOOP_LOG_CB
    if _LLAMA_NOOP_LOG_CB is not None:
        return
    try:
        import llama_cpp  # type: ignore[import-untyped]

        log_callback_type = ctypes.CFUNCTYPE(
            None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
        )

        @log_callback_type
        def _noop_cb(_level: int, _text: bytes, _user_data: int) -> None:
            return

        _LLAMA_NOOP_LOG_CB = _noop_cb
        llama_cpp.llama_log_set(_LLAMA_NOOP_LOG_CB, ctypes.c_void_p())
    except Exception:
        pass


def escape_gbnf_literal(text: str) -> str:
    """Escape a string literal for inclusion inside a GBNF rule."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def build_category_grammar_string(categories: Sequence[str]) -> str:
    """Construct a GBNF grammar constraining generation strictly to valid category names."""
    valid = [c.strip() for c in categories if c.strip()]
    if not valid:
        valid = list(DEFAULT_CATEGORIES)
    alternatives = " | ".join(escape_gbnf_literal(cat) for cat in valid)
    return f"root ::= {alternatives}\n"


_READ_SEARCH_TOOLS = frozenset(
    {
        "read_file",
        "search_web",
        "grep_search",
        "find_by_name",
        "list_dir",
        "view_file",
        "view_file_outline",
        "view_code_item",
        "read_url_content",
    }
)
_CODE_WRITE_TOOLS = frozenset(
    {
        "write_file",
        "edit_file",
        "multi_edit",
        "apply_patch",
        "replace_file_content",
        "multi_replace_file_content",
    }
)
_TASK_CMD_TOOLS = frozenset({"manage_task", "send_command_input", "command_status"})
_COMM_TOOLS = frozenset({"ask_question", "notify_user"})

_TEST_EXECUTABLES = frozenset(
    {
        "pytest",
        "py.test",
        "tox",
        "nox",
        "nosetests",
        "jest",
        "vitest",
        "mocha",
        "ava",
        "cypress",
        "playwright",
        "ruff",
        "flake8",
        "mypy",
        "pyright",
        "pylint",
        "eslint",
        "tsc",
        "biome",
        "stylelint",
        "shellcheck",
        "golangci-lint",
        "rspec",
        "phpunit",
        "ctest",
    }
)

_READONLY_EXECUTABLES = frozenset(
    {
        "ls",
        "ll",
        "la",
        "tree",
        "find",
        "fd",
        "fdfind",
        "rg",
        "ripgrep",
        "grep",
        "egrep",
        "fgrep",
        "ag",
        "ack",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "bat",
        "wc",
        "stat",
        "file",
        "du",
        "df",
        "which",
        "whereis",
        "type",
        "uname",
        "whoami",
        "id",
        "printenv",
        "ps",
        "lsof",
        "jq",
        "yq",
        "diff",
        "cmp",
        "md5",
        "shasum",
        "sha256sum",
        "pwd",
        "realpath",
        "dirname",
        "basename",
        "readlink",
        "sort",
        "uniq",
        "cut",
        "tr",
        "awk",
        "sed",
    }
)

_GIT_READONLY_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "rev-parse",
        "ls-files",
        "grep",
        "blame",
        "shortlog",
        "describe",
    }
)

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Detect output redirection (`>`, `>>`) that is NOT targeting /dev/null or fd duplication (`>&2`)
_FILE_REDIRECT_RE = re.compile(r"(?<![0-9&])>>?\s*(?![&/]*dev/null\b)(?![&][0-9])\S+")


def _split_shell_segments(command_line: str) -> list[str]:
    """Split a shell command string on unquoted `&&`, `||`, `;`, `|`, or newlines."""
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    n = len(command_line)
    while i < n:
        ch = command_line[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            buf.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in (";", "\n"):
            seg = "".join(buf).strip()
            if seg:
                segments.append(seg)
            buf = []
            i += 1
            continue
        if ch in ("&", "|"):
            if i + 1 < n and command_line[i + 1] == ch:
                seg = "".join(buf).strip()
                if seg:
                    segments.append(seg)
                buf = []
                i += 2
                continue
            if ch == "|":
                seg = "".join(buf).strip()
                if seg:
                    segments.append(seg)
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _has_unquoted_file_redirect(segment: str) -> bool:
    """Check whether a shell segment redirects output to a file (ignoring quoted strings)."""
    stripped_quotes = re.sub(r"'[^']*'|\"[^\"]*\"", "''", segment)
    return bool(_FILE_REDIRECT_RE.search(stripped_quotes))


def _tokenize_segment(segment: str) -> list[str]:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()

    idx = 0
    n = len(tokens)
    while idx < n:
        tok = tokens[idx]
        if _ENV_ASSIGN_RE.match(tok):
            idx += 1
            continue
        base = PurePath(tok).name.lower()
        if base in ("env", "time", "nice", "nohup", "sudo", "command"):
            idx += 1
            continue
        if base == "timeout" and idx + 2 <= n:
            idx += 2
            continue
        break
    return tokens[idx:]


def _segment_domain(segment: str) -> str | None:
    """Classify one shell segment into `'testing'`, `'exploration'`, `'commands'`, or `None` (noop)."""
    tokens = _tokenize_segment(segment)
    if not tokens:
        return None

    exe = PurePath(tokens[0]).name.lower()
    args = tokens[1:]
    args_low = [a.lower() for a in args]

    # Pure directory navigation / no-op prefixes in compound chains (`cd repo && ...`)
    if exe in ("cd", "pushd", "popd", "true", ":"):
        return None

    # 1. Direct test / lint / typecheck tools
    if exe in _TEST_EXECUTABLES:
        return "testing"

    # Wrapper runners (`uv run ...`, `poetry run ...`, `npx ...`, `bunx ...`)
    if exe in ("uv", "poetry", "pipenv") and args_low[:1] == ["run"] and len(args) >= 2:
        inner = " ".join(tokens[2:])
        return _segment_domain(inner) or "commands"
    if exe in ("npx", "bunx", "pnpx") and args:
        inner = " ".join(tokens[1:])
        return _segment_domain(inner) or "commands"

    # Python module test runners (`python -m pytest`, `python -m unittest`, `python -m mypy`)
    if exe in ("python", "python3") or exe.startswith("python3."):
        if "-m" in args_low:
            m_idx = args_low.index("-m")
            if m_idx + 1 < len(args_low):
                mod = args_low[m_idx + 1]
                if mod in ("pytest", "unittest", "mypy", "ruff", "flake8", "pylint", "doctest", "coverage"):
                    return "testing"
        if any(a.endswith(("_test.py",)) or PurePath(a).name.startswith("test_") for a in args_low if not a.startswith("-")):
            return "testing"
        return "commands"

    # JS/TS/Rust/Go/Make/JVM/Dotnet test subcommands
    if exe in ("npm", "pnpm", "yarn", "bun"):
        sub = [a for a in args_low if not a.startswith("-")]
        if sub and (
            sub[0] in ("test", "t", "tst", "lint", "typecheck", "check")
            or (sub[0] == "run" and len(sub) >= 2 and any(k in sub[1] for k in ("test", "lint", "typecheck", "check")))
        ):
            return "testing"
        return "commands"

    if exe == "cargo":
        sub = [a for a in args_low if not a.startswith("-")]
        if sub and sub[0] in ("test", "check", "clippy", "nextest"):
            return "testing"
        return "commands"

    if exe == "go":
        sub = [a for a in args_low if not a.startswith("-")]
        if sub and sub[0] in ("test", "vet"):
            return "testing"
        return "commands"

    if exe in ("make", "just"):
        sub = [a for a in args_low if not a.startswith("-")]
        if sub and any(k in sub[0] for k in ("test", "check", "lint", "verify")):
            return "testing"
        return "commands"

    if exe in ("gradle", "gradlew", "mvn", "mvnw", "dotnet"):
        sub = [a for a in args_low if not a.startswith("-")]
        if sub and sub[0] in ("test", "check", "verify"):
            return "testing"
        return "commands"

    # 2. Mutating output redirection turns otherwise read-only tools into `commands`
    if _has_unquoted_file_redirect(segment):
        return "commands"

    # 3. Read-only inspection tools
    if exe in _READONLY_EXECUTABLES:
        if exe == "find" and any(f in args_low for f in ("-delete", "-exec", "-execdir", "-ok")):
            return "commands"
        if exe == "sed" and any(a == "-i" or a.startswith("-i") or a == "--in-place" for a in args_low):
            return "commands"
        return "exploration"

    if exe == "git":
        # Strip global git flags like `-C <dir>` or `--no-pager`
        idx = 0
        while idx < len(args_low):
            tok = args_low[idx]
            if tok in ("-c", "-C", "--git-dir", "--work-tree") and idx + 1 < len(args_low):
                idx += 2
                continue
            if tok.startswith("-"):
                idx += 1
                continue
            break
        if idx >= len(args_low):
            return "exploration"
        subcmd = args_low[idx]
        rest = args_low[idx + 1 :]
        if subcmd in _GIT_READONLY_SUBCOMMANDS:
            return "exploration"
        if subcmd == "branch" and not any(f in rest for f in ("-d", "-D", "--delete", "-m", "-M", "-c", "-C")):
            return "exploration"
        if subcmd == "remote" and (not rest or rest == ["-v"] or rest == ["--verbose"]):
            return "exploration"
        if subcmd == "tag" and (not rest or rest[0] in ("-l", "--list")):
            return "exploration"
        if subcmd == "config" and any(f in rest for f in ("--get", "--get-all", "--list", "-l")):
            return "exploration"
        return "commands"

    return "commands"


def classify_shell_command(command_line: str) -> str:
    """
    Classify a `run_command` shell string into one of three semantic domains:
    - `'testing'`: Running test suites, linters, or typecheckers (`pytest`, `npm test`, `mypy`, ...)
    - `'exploration'`: Read-only repo/system inspection (`ls`, `find`, `rg`, `cat`, `git status`/`diff`, ...)
    - `'commands'`: Builds, installs, scripts, file/git mutations, servers, or general shell execution
    """
    raw = (command_line or "").strip()
    if not raw:
        return "commands"

    segments = _split_shell_segments(raw)
    domains = [d for seg in segments if (d := _segment_domain(seg)) is not None]
    if not domains:
        return "exploration"
    if "testing" in domains:
        return "testing"
    if all(d == "exploration" for d in domains):
        return "exploration"
    return "commands"


def _classify_tool_domain(tc: ToolCallSummary) -> str | None:
    """Map a single tool invocation to a semantic domain, or `None` if context-dependent."""
    t_low = tc.name.lower()
    summary_low = tc.summary.lower()

    if t_low in _READ_SEARCH_TOOLS:
        return "exploration"
    if t_low in _CODE_WRITE_TOOLS:
        if "implementation_plan" in summary_low:
            return "planning"
        return "coding"
    if t_low == "run_command":
        return classify_shell_command(tc.summary)
    if t_low in _TASK_CMD_TOOLS:
        return "commands"
    if t_low in _COMM_TOOLS:
        return "communication"
    return None


def _match_domain_to_category(domain: str, categories: Sequence[str]) -> str | None:
    """Resolve a semantic domain to a configured category name, or `None` if unmapped."""
    cats = list(categories)
    lower_map = {c.lower(): c for c in cats}

    # Direct canonical match
    if domain in lower_map:
        return lower_map[domain]

    if domain == "planning":
        keywords = ("plan", "design", "arch")
    elif domain == "exploration":
        keywords = ("explor", "research", "search", "read", "inspect", "investigat")
    elif domain == "coding":
        keywords = ("cod", "implement", "edit", "write", "dev")
    elif domain == "testing":
        keywords = ("test", "verif", "qa", "lint", "check", "exec")
    elif domain == "commands":
        # Prefer explicit command/shell/build categories before legacy `Execution/Testing`
        for kw in ("command", "cmd", "shell", "ops", "build", "script"):
            for key, orig in lower_map.items():
                if kw in key:
                    return orig
        for key, orig in lower_map.items():
            if "exec" in key:
                return orig
        return None
    elif domain == "communication":
        keywords = ("comm", "chat", "reply", "resp", "summar", "review")
    else:
        return None

    for kw in keywords:
        for key, orig in lower_map.items():
            if kw in key:
                return orig
    return None


_PROPOSED_PLAN_RE = re.compile(
    r"\b(?:implementation plan|proposed plan|here is (?:the|my) plan|step-by-step plan|"
    r"plan of attack|before (?:i|we) (?:proceed|implement|start)|shall i proceed)\b",
    re.IGNORECASE,
)


_OVERHEAD_CATEGORY_NAMES = frozenset(
    {"harness overhead", "system overhead", "system setup"}
)


def resolve_tool_guardrail(
    step: StepRecord,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> tuple[str | None, tuple[str, ...]]:
    """
    Apply structural tool-signature guardrails prior to local LLM inference.

    Returns `(deterministic_category, candidate_categories)`:
    - When a step invokes tools belonging to a single unambiguous domain mapped in `categories`,
      returns `(category, (category,))` immediately so a small LLM cannot misclassify e.g.
      `read_file(tests/test_foo.py)` as `Testing` or `ls -la` as `Commands`.
    - When a step invokes tools from multiple mapped domains, restricts `candidate_categories`
      to those domains for GBNF-constrained LLM disambiguation.
    - When no tools are invoked, resolves pure reasoning turns to `Planning`, resolves direct
      user explanations/summaries (without plan-proposal cues) to `Communication`, and restricts
      ambiguous plan-vs-reply turns to `(Planning, Communication)`.
    """
    raw_cats = tuple(categories) if categories else DEFAULT_CATEGORIES
    filtered = tuple(c for c in raw_cats if c.strip().lower() not in _OVERHEAD_CATEGORY_NAMES)
    cats = filtered if filtered else raw_cats

    if step.tool_calls:
        mapped: list[str] = []
        for tc in step.tool_calls:
            dom = _classify_tool_domain(tc)
            if dom is None:
                return None, cats
            cat = _match_domain_to_category(dom, cats)
            if cat is None:
                return None, cats
            if cat not in mapped:
                mapped.append(cat)

        if len(mapped) == 1:
            return mapped[0], (mapped[0],)
        if len(mapped) > 1:
            return None, tuple(mapped)
        return None, cats

    # No tools invoked in this step
    plan_cat = _match_domain_to_category("planning", cats)
    comm_cat = _match_domain_to_category("communication", cats)

    if not step.content.strip():
        if plan_cat is not None:
            return plan_cat, (plan_cat,)
        return None, cats

    if plan_cat is not None and comm_cat is not None and plan_cat != comm_cat:
        if _PROPOSED_PLAN_RE.search(step.content):
            return None, (plan_cat, comm_cat)
        return comm_cat, (comm_cat,)
    return None, cats


def _compact_summary_for_prompt(tool_name: str, summary: str) -> str:
    """Shorten long absolute file paths so filenames stay visible in the classifier prompt."""
    if not summary:
        return ""
    if tool_name.lower() in (_READ_SEARCH_TOOLS | _CODE_WRITE_TOOLS) and summary.startswith("/"):
        parts = [p for p in summary.split("/") if p]
        if len(parts) > 3:
            return ".../" + "/".join(parts[-3:])
    return summary[:120]


def format_step_for_classifier(step: StepRecord, max_chars: int = 1600) -> str:
    """Condense an agent PLANNER_RESPONSE step into a compact classification text block."""
    parts: list[str] = []
    if step.thinking_tokens > 0:
        parts.append(
            f"Output tokens: {step.output_tokens} "
            f"(thinking: {step.thinking_tokens}, visible: {step.visible_output_tokens})"
        )
    if step.tool_calls:
        rendered_tools = []
        domains: list[str] = []
        for tc in step.tool_calls[:8]:
            dom = _classify_tool_domain(tc)
            if dom == "exploration" and "exploration: read/inspect" not in domains:
                domains.append("exploration: read/inspect")
            elif dom == "coding" and "coding: write/edit" not in domains:
                domains.append("coding: write/edit")
            elif dom == "testing" and "testing: test/lint" not in domains:
                domains.append("testing: test/lint")
            elif dom == "commands" and "commands: shell/task" not in domains:
                domains.append("commands: shell/task")
            elif dom == "planning" and "planning: plan artifact" not in domains:
                domains.append("planning: plan artifact")
            elif dom == "communication" and "communication: ask user" not in domains:
                domains.append("communication: ask user")

            compact_arg = _compact_summary_for_prompt(tc.name, tc.summary)
            if compact_arg:
                rendered_tools.append(f"{tc.name}({compact_arg})")
            else:
                rendered_tools.append(tc.name)
        domain_hint = f" [{', '.join(domains)}]" if domains else ""
        parts.append(f"Tools invoked{domain_hint}: " + ", ".join(rendered_tools))

        if step.thinking.strip():
            snippet = " ".join(step.thinking.strip().split())
            parts.append(f"Reasoning: {snippet[:650]}")
        if step.content.strip():
            snippet = " ".join(step.content.strip().split())
            parts.append(f"Message to user: {snippet[:700]}")
    else:
        parts.append("Tools invoked: none (no tools called)")
        # Place user-facing message first on no-tool turns so replies aren't masked by internal thoughts
        if step.content.strip():
            snippet = " ".join(step.content.strip().split())
            parts.append(f"Message to user: {snippet[:850]}")
        if step.thinking.strip():
            snippet = " ".join(step.thinking.strip().split())
            parts.append(f"Reasoning: {snippet[:550]}")

    combined = "\n".join(parts).strip()
    return combined[:max_chars]


def heuristic_classify(
    step: StepRecord,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
) -> str:
    """Deterministic classifier used when heuristic-only mode is enabled or LLM inference is unavailable."""
    cats = list(categories) if categories else list(DEFAULT_CATEGORIES)

    direct, candidates = resolve_tool_guardrail(step, cats)
    if direct is not None:
        return direct

    # Mixed-tool turn: prioritize Coding > Testing > Commands > Exploration > Planning > Communication
    if step.tool_calls and candidates:
        for preferred_dom in ("coding", "testing", "commands", "exploration", "planning", "communication"):
            matched = _match_domain_to_category(preferred_dom, candidates)
            if matched is not None:
                return matched
        return candidates[0]

    # Subagent / custom tool calls
    if step.tool_calls:
        tool_names = [tc.name.lower() for tc in step.tool_calls]
        if any(t == "invoke_subagent" for t in tool_names):
            return _match_domain_to_category("exploration", cats) or cats[0]

    # No-tool turn with user-facing message: Communication unless explicitly proposing a plan
    if step.content.strip():
        if _PROPOSED_PLAN_RE.search(step.content):
            return _match_domain_to_category("planning", cats) or cats[0]
        return _match_domain_to_category("communication", cats) or cats[0]

    return _match_domain_to_category("planning", cats) or cats[0]


def ensure_model_weights(
    model_spec: ModelSpec,
    model_dir: Path,
    hf_token: str | None = None,
    status_callback: Callable[[str], None] | None = None,
    downloader: Callable[..., str] | None = None,
    show_progress: bool = False,
) -> Path:
    """
    Ensure GGUF model weights exist on disk, reading from disk first before calling Hugging Face API.

    Authentication uses `hf_token` (resolved from HF_TOKEN / HUGGING_FACE_HUB_TOKEN).
    """
    target_path = model_spec.disk_target(model_dir)
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target_path

    if model_spec.repo_id == "local":
        raise FileNotFoundError(f"Local GGUF model file does not exist: {target_path}")

    token = hf_token if hf_token is not None else get_hf_token()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    using_default_hf = downloader is None
    if downloader is None:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

        downloader = hf_hub_download

    # 1. Check standard local Hugging Face cache without network I/O
    try:
        cached_file = downloader(
            repo_id=model_spec.repo_id,
            filename=model_spec.filename,
            token=token,
            local_files_only=True,
        )
        cached_path = Path(cached_file)
        if cached_path.is_file() and cached_path.stat().st_size > 0:
            if cached_path.resolve() != target_path.resolve():
                shutil.copy2(cached_path, target_path)
            return target_path
    except Exception:
        pass

    # 2. Download from Hugging Face Hub and persist directly to target_path.parent on disk
    if status_callback is not None:
        status_callback(f"downloading {model_spec.name}")

    re_enable_bars = False
    if using_default_hf and not show_progress:
        try:
            from huggingface_hub.utils import (  # type: ignore[import-untyped]
                are_progress_bars_disabled,
                disable_progress_bars,
            )

            if not are_progress_bars_disabled():
                disable_progress_bars()
                re_enable_bars = True
        except Exception:
            pass

    try:
        downloaded = downloader(
            repo_id=model_spec.repo_id,
            filename=model_spec.filename,
            local_dir=str(target_path.parent),
            token=token,
        )
    finally:
        if re_enable_bars:
            try:
                from huggingface_hub.utils import enable_progress_bars  # type: ignore[import-untyped]

                enable_progress_bars()
            except Exception:
                pass
    downloaded_path = Path(downloaded)
    if downloaded_path.is_file() and downloaded_path.resolve() != target_path.resolve():
        shutil.copy2(downloaded_path, target_path)

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Downloaded weights from {model_spec.repo_id}/{model_spec.filename}, "
            f"but expected file not found at {target_path}"
        )
    return target_path


class StepClassifierBackend(Protocol):
    """Protocol implemented by OpenModelRouter and test doubles."""

    def load(self, status_callback: Callable[[str], None] | None = None) -> None: ...

    def classify(self, step: StepRecord) -> str: ...


CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Planning": "Proposing an upcoming implementation plan/roadmap before making changes, or internal reasoning with no message to user.",
    "Exploration": "Reading/searching files (read_file, grep_search), read-only shell inspection (ls, find, rg, git status/diff), or web search.",
    "Coding": "Creating, editing, or patching files (write_file, edit_file, multi_edit, apply_patch).",
    "Commands": "Running non-test shell commands, builds, installs, scripts, git mutations, or background tasks (run_command, manage_task).",
    "Testing": "Running test suites, linters, or typecheckers (pytest, npm test, cargo test, mypy, ruff).",
    "Execution/Testing": "Running shell/terminal commands or test suites (run_command, manage_task).",
    "Communication": "Replying to the user to explain architecture/code, answer a question, summarize completed work, or ask a question.",
}


class OpenModelRouter:
    """
    Manages disk caching, loading, tool-signature guardrails, and GBNF-constrained
    inference of a configurable local open-weight GGUF model via huggingface_hub + llama-cpp-python.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        model_dir: Path,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
        hf_token: str | None = None,
        n_gpu_layers: int = -1,
        n_threads: int = 4,
        llm_factory: Callable[..., Any] | None = None,
        downloader: Callable[..., str] | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.model_dir = model_dir
        self.categories = tuple(categories) if categories else DEFAULT_CATEGORIES
        self.hf_token = hf_token if hf_token is not None else get_hf_token()
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._llm_factory = llm_factory
        self._downloader = downloader

        self._lock = threading.Lock()
        self._llm: Any = None
        self._grammar: Any = None
        self._grammar_cache: dict[tuple[str, ...], Any] = {}
        self._weights_path: Path | None = None

    @property
    def weights_path(self) -> Path | None:
        return self._weights_path

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self, status_callback: Callable[[str], None] | None = None) -> None:
        """Ensure weights are saved on disk and initialize the local llama.cpp model."""
        with self._lock:
            if self._llm is not None:
                return

            weights_path = ensure_model_weights(
                model_spec=self.model_spec,
                model_dir=self.model_dir,
                hf_token=self.hf_token,
                status_callback=status_callback,
                downloader=self._downloader,
            )
            self._weights_path = weights_path

            if status_callback is not None:
                status_callback(f"loading {self.model_spec.name}")

            _silence_llama_cpp_logs()

            if self._llm_factory is not None:
                self._llm = self._llm_factory(
                    model_path=str(weights_path),
                    n_ctx=self.model_spec.context_length,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=False,
                )
            else:
                from llama_cpp import Llama, LlamaGrammar  # type: ignore[import-untyped]

                grammar_text = build_category_grammar_string(self.categories)
                self._grammar = LlamaGrammar.from_string(grammar_text, verbose=False)
                self._grammar_cache[self.categories] = self._grammar
                self._llm = Llama(
                    model_path=str(weights_path),
                    n_ctx=self.model_spec.context_length,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    verbose=False,
                )

            if status_callback is not None:
                status_callback(self.model_spec.name)

    def _get_grammar_for(self, candidates: tuple[str, ...]) -> Any:
        if self._grammar is None:
            return None
        if candidates in self._grammar_cache:
            return self._grammar_cache[candidates]
        try:
            from llama_cpp import LlamaGrammar  # type: ignore[import-untyped]

            grammar = LlamaGrammar.from_string(
                build_category_grammar_string(candidates),
                verbose=False,
            )
            self._grammar_cache[candidates] = grammar
            return grammar
        except Exception:
            return self._grammar

    def _build_messages(
        self,
        step_summary: str,
        candidates: Sequence[str],
    ) -> list[dict[str, str]]:
        cat_list = ", ".join(candidates)
        lines = [
            f"Classify the agent turn into EXACTLY ONE label: {cat_list}.",
            "Prioritize the concrete action taken in `Tools invoked` (or `Message to user` when no tools are called) over background topics discussed in `Reasoning`.",
        ]
        for cat in candidates:
            desc = CATEGORY_DESCRIPTIONS.get(cat, f"Activity related to {cat}.")
            lines.append(f"- {cat}: {desc}")
        sys_prompt = "\n".join(lines)

        messages: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
        cat_set = set(candidates)
        few_shot: list[tuple[str, str]] = [
            (
                "Tools invoked: none (no tools called)\nMessage to user: Proposed implementation plan:\n1. Update state.py\n2. Add tests\nShall I proceed?",
                "Planning",
            ),
            (
                "Tools invoked: none (no tools called)\nMessage to user: `agy-token-viz` is a CLI wrapper that tails transcripts and renders a 2-line bar. Here is how each module works...",
                "Communication",
            ),
            (
                "Tools invoked: none (no tools called)\nMessage to user: Updated README.md with setup instructions, design decisions, and limitations.",
                "Communication",
            ),
            (
                "Tools invoked [coding: write/edit]: write_file(src/app.py), edit_file(src/cli.py)\nReasoning: Editing source code.",
                "Coding",
            ),
            (
                "Tools invoked [commands: shell/task]: run_command(npm run build)\nReasoning: Building the package.",
                "Commands",
            ),
            (
                "Tools invoked [testing: test/lint]: run_command(pytest -q)\nReasoning: Running test suite.",
                "Testing",
            ),
            (
                "Tools invoked [exploration: read/inspect]: read_file(docs/guide.md), run_command(ls -la)\nReasoning: Inspecting files.",
                "Exploration",
            ),
        ]
        for example_input, label in few_shot:
            if label in cat_set:
                messages.append({"role": "user", "content": f"Log:\n{example_input}\nLabel:"})
                messages.append({"role": "assistant", "content": label})

        messages.append({"role": "user", "content": f"Log:\n{step_summary}\nLabel:"})
        return messages

    def classify(self, step: StepRecord) -> str:
        """Classify an agent step using structural tool guardrails + GBNF-constrained local LLM inference."""
        direct, candidates = resolve_tool_guardrail(step, self.categories)
        if direct is not None:
            return direct

        if self._llm is None:
            self.load()

        active_candidates = candidates if candidates else self.categories
        step_summary = format_step_for_classifier(step)
        messages = self._build_messages(step_summary, active_candidates)

        with self._lock:
            grammar = self._get_grammar_for(active_candidates)
            kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16,
            }
            if grammar is not None:
                kwargs["grammar"] = grammar

            try:
                response = self._llm.create_chat_completion(**kwargs)
                raw = (
                    response["choices"][0]["message"]["content"]
                    if isinstance(response, dict)
                    else str(response)
                )
            except Exception:
                sys_prompt = messages[0]["content"]
                prompt = f"{sys_prompt}\n\nTurn:\n{step_summary}\n\nCategory: "
                comp_kwargs: dict[str, Any] = {
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": 16,
                }
                if grammar is not None:
                    comp_kwargs["grammar"] = grammar
                comp = self._llm.create_completion(**comp_kwargs)
                raw = comp["choices"][0]["text"]

        cleaned = (raw or "").strip()
        lowered = cleaned.lower()
        for cat in active_candidates:
            if cat.lower() == lowered:
                return cat
        for cat in active_candidates:
            if cat.lower() in lowered:
                return cat
        return heuristic_classify(step, active_candidates)

    @classmethod
    def from_config(cls, config: VizConfig) -> "OpenModelRouter":
        return cls(
            model_spec=config.model_spec,
            model_dir=config.model_dir,
            categories=config.categories,
            hf_token=config.hf_token,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
        )


class AsyncStepClassifier:
    """Background worker that warms up OpenModelRouter and classifies steps asynchronously."""

    def __init__(
        self,
        state: SessionTokenState,
        router: StepClassifierBackend | None,
        heuristic_only: bool = False,
    ) -> None:
        self._state = state
        self._router = router
        self._heuristic_only = heuristic_only or (router is None)
        self._fallback_mode = self._heuristic_only
        self._queue: queue.Queue[StepRecord | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="agy-token-viz-router",
            daemon=True,
        )
        self._thread.start()

    def submit(self, step: StepRecord) -> None:
        """Register step in the ledger immediately and queue for classification."""
        if self._state.register_step(step):
            self._queue.put(step)

    def stop(self, timeout: float = 3.0) -> None:
        """Signal worker to finish queued steps and exit cleanly."""
        if not self._started:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        if self._heuristic_only or self._router is None:
            self._state.set_router_status("heuristic")
        else:
            try:
                self._router.load(status_callback=self._state.set_router_status)
            except Exception as exc:
                logger.debug("Router load failed, falling back to heuristic: %s", exc)
                self._fallback_mode = True
                self._state.set_router_status("heuristic (offline)")

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                if self._fallback_mode or self._router is None:
                    category = heuristic_classify(item, self._state.categories)
                else:
                    category = self._router.classify(item)
            except Exception as exc:
                logger.debug("Classification error on step %s: %s", item.step_index, exc)
                category = heuristic_classify(item, self._state.categories)

            self._state.complete_classification(
                item.conversation_id,
                item.step_index,
                category,
            )
            self._queue.task_done()
