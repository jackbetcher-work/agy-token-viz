"""Unit tests for OpenModelRouter, shell command classification, guardrails, and GBNF (`router.py`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_cpp import LlamaGrammar

from agy_token_viz.config import ModelSpec, get_hf_token, resolve_model_spec
from agy_token_viz.router import (
    AsyncStepClassifier,
    OpenModelRouter,
    build_category_grammar_string,
    classify_shell_command,
    ensure_model_weights,
    format_step_for_classifier,
    heuristic_classify,
    resolve_tool_guardrail,
)
from agy_token_viz.state import SessionTokenState, StepRecord, ToolCallSummary


def test_resolve_model_spec_presets_and_custom(tmp_path: Path) -> None:
    default_spec = resolve_model_spec()
    assert default_spec.name == "qwen2.5-3b"
    assert default_spec.repo_id == "Qwen/Qwen2.5-3B-Instruct-GGUF"

    preset_7b = resolve_model_spec("qwen2.5-7b")
    assert preset_7b.repo_id == "bartowski/Qwen2.5-7B-Instruct-GGUF"

    preset = resolve_model_spec("qwen2.5-0.5b")
    assert preset.repo_id == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"

    custom = resolve_model_spec("my-org/my-router-gguf:router-q4_k_m.gguf")
    assert custom.repo_id == "my-org/my-router-gguf"
    assert custom.filename == "router-q4_k_m.gguf"

    local_file = tmp_path / "weights.gguf"
    local_file.write_bytes(b"GGUF")
    local_spec = resolve_model_spec(str(local_file))
    assert local_spec.repo_id == "local"
    assert local_spec.disk_target(tmp_path) == local_file.resolve()


def test_get_hf_token_reads_env_vars() -> None:
    assert get_hf_token({"HF_TOKEN": "hf_secret_123"}) == "hf_secret_123"
    assert get_hf_token({"HUGGING_FACE_HUB_TOKEN": "hf_alt_456"}) == "hf_alt_456"
    assert get_hf_token({}) is None


def test_ensure_model_weights_reads_from_disk_before_network(tmp_path: Path) -> None:
    spec = ModelSpec(
        name="test-model",
        repo_id="org/test-gguf",
        filename="model.gguf",
    )
    calls: list[dict[str, Any]] = []

    def _fake_downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not in hf cache")
        out_dir = Path(kwargs["local_dir"])
        out_file = out_dir / kwargs["filename"]
        out_file.write_bytes(b"FAKE-GGUF-WEIGHTS")
        return str(out_file)

    # 1st call downloads via HF API with token and saves to disk
    path1 = ensure_model_weights(
        model_spec=spec,
        model_dir=tmp_path,
        hf_token="hf_test_token",
        downloader=_fake_downloader,
    )
    assert path1.is_file()
    assert path1.read_bytes() == b"FAKE-GGUF-WEIGHTS"
    assert any(c.get("token") == "hf_test_token" and not c.get("local_files_only") for c in calls)

    # 2nd call finds weights on disk immediately -> 0 additional downloader calls!
    call_count_before = len(calls)
    path2 = ensure_model_weights(
        model_spec=spec,
        model_dir=tmp_path,
        hf_token="hf_test_token",
        downloader=_fake_downloader,
    )
    assert path2 == path1
    assert len(calls) == call_count_before


def test_gbnf_grammar_compiles_with_llama_cpp() -> None:
    categories = ("Planning", "Exploration", "Coding", "Commands", "Testing", 'Custom "Quote"')
    grammar_str = build_category_grammar_string(categories)
    grammar = LlamaGrammar.from_string(grammar_str, verbose=False)
    assert grammar is not None
    assert '"Commands"' in grammar_str
    assert '"Testing"' in grammar_str


def test_classify_shell_command_separates_exploration_commands_and_testing() -> None:
    # Read-only exploration commands
    assert classify_shell_command("ls -la /Users/jackbetcher/project") == "exploration"
    assert classify_shell_command("find . -maxdepth 4 -not -path '*/.*'") == "exploration"
    assert classify_shell_command("rg 'SessionTokenState' src/ | head -n 20") == "exploration"
    assert classify_shell_command("git status && git diff --stat") == "exploration"

    # Mutating shell commands / scripts / installs
    assert classify_shell_command("find . -name '*.pyc' -delete") == "commands"
    assert classify_shell_command("sed -i '' 's/foo/bar/g' a.txt") == "commands"
    assert classify_shell_command("cat template.md > README.md") == "commands"
    assert classify_shell_command(".venv/bin/pip install -e .") == "commands"
    assert classify_shell_command("npm run build") == "commands"
    assert classify_shell_command("git commit -m 'init'") == "commands"

    # Test / lint / verification commands
    assert classify_shell_command(".venv/bin/pytest -q") == "testing"
    assert classify_shell_command("PYTHONPATH=src python3 -m pytest tests/") == "testing"
    assert classify_shell_command("cd repo && npm test") == "testing"
    assert classify_shell_command("cargo clippy --all-targets") == "testing"
    assert classify_shell_command("uv run mypy src") == "testing"


def test_tool_guardrails_prevent_misclassification_of_test_file_reads_and_replies() -> None:
    # Reading test files must ALWAYS be Exploration, never Testing
    read_test_step = StepRecord(
        "c1",
        1,
        1000,
        0,
        200,
        thinking="Inspecting pytest unit tests",
        tool_calls=(ToolCallSummary("read_file", "/repo/tests/test_router.py"),),
    )
    direct, _ = resolve_tool_guardrail(read_test_step)
    assert direct == "Exploration"

    # Direct explanation to user without tools must be Communication (even if mentioning tests/runner)
    reply_step = StepRecord(
        "c1",
        2,
        1000,
        0,
        500,
        thinking="Reviewing PTY runner and test suite details.",
        content="Here is how the repository architecture and test suite are structured.",
    )
    direct_reply, _ = resolve_tool_guardrail(reply_step)
    assert direct_reply == "Communication"

    # Proposed implementation plan with no tools restricts candidates to (Planning, Communication)
    plan_step = StepRecord(
        "c1",
        3,
        1000,
        0,
        400,
        content="Proposed plan:\n1. Edit state.py\n2. Run tests\nShall I proceed?",
    )
    direct_plan, cands = resolve_tool_guardrail(plan_step)
    assert direct_plan is None
    assert cands == ("Planning", "Communication")


def test_open_model_router_and_async_worker_classify_steps(tmp_path: Path) -> None:
    local_gguf = tmp_path / "dummy.gguf"
    local_gguf.write_bytes(b"DUMMY")
    spec = resolve_model_spec(str(local_gguf))
    llm_calls: list[dict[str, Any]] = []

    class _FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            llm_calls.append(kwargs)
            user_msg = kwargs["messages"][-1]["content"]
            if "Proposed plan" in user_msg:
                label = "Planning"
            elif "edit_file" in user_msg:
                label = "Coding"
            else:
                label = "Communication"
            return {"choices": [{"message": {"content": label}}]}

    router = OpenModelRouter(
        model_spec=spec,
        model_dir=tmp_path,
        llm_factory=lambda **kw: _FakeLlama(**kw),
    )
    state = SessionTokenState()
    worker = AsyncStepClassifier(state=state, router=router)
    worker.start()

    # Step 1: Mixed read_file + edit_file -> invokes LLM disambiguation -> Coding
    worker.submit(
        StepRecord(
            conversation_id="c1",
            step_index=1,
            input_tokens=2000,
            cache_read_tokens=0,
            output_tokens=300,
            tool_calls=(
                ToolCallSummary("read_file", "src/app.py"),
                ToolCallSummary("edit_file", "src/app.py"),
            ),
            preceded_by_user_input=True,
        )
    )
    # Step 2: Single-domain pytest -> guardrail resolves Testing; 1200 tool-result tokens go to Step 1
    worker.submit(
        StepRecord(
            conversation_id="c1",
            step_index=2,
            input_tokens=1500,
            cache_read_tokens=2000,
            output_tokens=200,
            tool_calls=(ToolCallSummary("run_command", "pytest -q"),),
        )
    )
    # Step 3: Ambiguous plan proposal -> invokes LLM restricted to (Planning, Communication) -> Planning;
    # 100 pytest-output tokens go to Step 2
    worker.submit(
        StepRecord(
            conversation_id="c1",
            step_index=3,
            input_tokens=300,
            cache_read_tokens=3500,
            output_tokens=400,
            content="Proposed plan:\n1. Refactor module\nShall I proceed?",
        )
    )
    worker.stop(timeout=3.0)

    snap = state.snapshot()
    by_name = {c.name: c for c in snap.categories}
    assert by_name["Coding"].active_tokens == 2000 + 300 + 1200
    assert by_name["Testing"].active_tokens == 200 + 100
    assert by_name["Planning"].active_tokens == 400
    assert snap.classified_steps == 3
    assert len(llm_calls) == 2


def test_heuristic_classify_covers_all_six_default_categories() -> None:
    assert (
        heuristic_classify(
            StepRecord("c", 1, 100, 0, 50, tool_calls=(ToolCallSummary("read_file", "a.py"),))
        )
        == "Exploration"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 2, 100, 0, 50, tool_calls=(ToolCallSummary("run_command", "ls -la"),))
        )
        == "Exploration"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 3, 100, 0, 50, tool_calls=(ToolCallSummary("edit_file", "a.py"),))
        )
        == "Coding"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 4, 100, 0, 50, tool_calls=(ToolCallSummary("run_command", "npm run build"),))
        )
        == "Commands"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 5, 100, 0, 50, tool_calls=(ToolCallSummary("run_command", "pytest -q"),))
        )
        == "Testing"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 6, 100, 0, 50, content="Updated the configuration file as requested.")
        )
        == "Communication"
    )
    assert (
        heuristic_classify(
            StepRecord("c", 7, 100, 0, 50, thinking="Breaking down the architecture first.")
        )
        == "Planning"
    )


def test_format_step_for_classifier_annotates_tool_domains_and_compacts_paths() -> None:
    text = format_step_for_classifier(
        StepRecord(
            "c",
            1,
            100,
            0,
            50,
            thinking="Checking docs",
            tool_calls=(
                ToolCallSummary("read_file", "/Users/dev/work/project/src/pkg/module.py"),
            ),
        )
    )
    assert "[exploration: read/inspect]" in text
    assert "read_file(.../src/pkg/module.py)" in text
