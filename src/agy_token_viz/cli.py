"""CLI entrypoint for `agy-token-viz` and `jetski-token-viz` (invoked directly or via `bin/agy` / `bin/jetski`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from agy_token_viz.config import (
    HARNESS_PROFILES,
    MODEL_PRESETS,
    VizConfig,
    discover_real_cli_binary,
)
from agy_token_viz.pty_runner import PTYVisualizerRunner
from agy_token_viz.renderer import render_final_summary, should_use_color
from agy_token_viz.router import (
    AsyncStepClassifier,
    OpenModelRouter,
    ensure_model_weights,
)
from agy_token_viz.state import SessionTokenState
from agy_token_viz.transcript_watcher import (
    TranscriptWatcher,
    extract_log_file_from_args,
    extract_target_conversation_from_args,
    is_resuming_conversation_from_args,
)


def parse_cli_args(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """
    Split visualizer arguments from forwarded `agy` / `jetski` CLI arguments.

    Supports both:
      agy-token-viz [viz-options] -- [cli-args...]
      jetski-token-viz [viz-options] -- [cli-args...]
    and mixed invocation from `bin/agy` or `bin/jetski`.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        sep = raw.index("--")
        viz_part = raw[:sep]
        cli_part = raw[sep + 1 :]
    else:
        viz_part = raw
        cli_part = []

    resolved_prog = prog or (Path(sys.argv[0]).name if sys.argv else "agy-token-viz")
    parser = argparse.ArgumentParser(
        prog=resolved_prog,
        description=(
            "Run Antigravity CLI (agy) or Jetski CLI (jetski) with a live 2-line tqdm-style "
            "token usage visualizer classified by activity category using a local open-weight model."
        ),
    )
    parser.add_argument(
        "--visualize-tokens",
        action="store_true",
        default=True,
        help="Enable live token visualization (enabled by default when invoking token-viz directly).",
    )
    parser.add_argument(
        "--viz-target",
        choices=tuple(HARNESS_PROFILES),
        default=None,
        help="Target CLI harness ('agy' or 'jetski'; inferred from invocation name by default).",
    )
    parser.add_argument(
        "--viz-data-dir",
        default=None,
        help="Explicit CLI app data directory override (or TOKEN_VIZ_DATA_DIR / AGY_DATA_DIR / JETSKI_DATA_DIR).",
    )
    parser.add_argument(
        "--viz-model",
        default=None,
        help=(
            f"Local classifier model preset ({', '.join(MODEL_PRESETS)}), local .gguf path, "
            "or '<hf_org>/<hf_repo>:<file.gguf>'. Also configurable via AGY_VIZ_MODEL / JETSKI_VIZ_MODEL."
        ),
    )
    parser.add_argument(
        "--viz-model-file",
        default=None,
        help="Optional GGUF filename override within the Hugging Face repo (or AGY_VIZ_MODEL_FILE).",
    )
    parser.add_argument(
        "--viz-model-dir",
        default=None,
        help="Local directory to cache/read Hugging Face GGUF weights (default: ~/.cache/agy-token-viz/models).",
    )
    parser.add_argument(
        "--viz-categories",
        default=None,
        help="Comma-separated list of activity categories (or AGY_VIZ_CATEGORIES).",
    )
    parser.add_argument(
        "--viz-config",
        default=None,
        help="Optional path to JSON configuration file (or AGY_VIZ_CONFIG).",
    )
    parser.add_argument(
        "--viz-include-history",
        action="store_true",
        help="Include prior steps when resuming a conversation via --continue or --conversation.",
    )
    parser.add_argument(
        "--viz-heuristic-only",
        action="store_true",
        help="Skip local LLM loading and classify steps with fast built-in rules.",
    )
    parser.add_argument(
        "--viz-preload-only",
        action="store_true",
        help="Download/verify local Hugging Face GGUF weights on disk, run a self-check, and exit.",
    )
    parser.add_argument(
        "--viz-bin",
        default=None,
        help="Explicit path to the underlying CLI binary (or TOKEN_VIZ_REAL_BIN).",
    )
    parser.add_argument(
        "--viz-agy-bin",
        default=None,
        help="Explicit path to the underlying `agy` binary (or set AGY_REAL_BIN).",
    )
    parser.add_argument(
        "--viz-jetski-bin",
        default=None,
        help="Explicit path to the underlying `jetski` binary (or set JETSKI_REAL_BIN).",
    )

    known, extra = parser.parse_known_args(viz_part)
    forwarded = [*extra, *cli_part]
    return known, forwarded


def _run_preload(config: VizConfig) -> int:
    tag = config.profile.tag
    sys.stderr.write(
        f"[{tag}] Checking local weights for '{config.model_spec.name}' "
        f"({config.model_spec.repo_id} / {config.model_spec.filename})...\n"
    )
    path = ensure_model_weights(
        model_spec=config.model_spec,
        model_dir=config.model_dir,
        hf_token=config.hf_token,
        status_callback=lambda msg: sys.stderr.write(f"[{tag}] {msg}...\n"),
        show_progress=True,
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    sys.stderr.write(
        f"[{tag}] Weights ready on disk: {path} ({size_mb:.1f} MB)\n"
    )

    router = OpenModelRouter.from_config(config)
    router.load()
    sys.stderr.write(
        f"[{tag}] Local router initialized successfully (categories: {', '.join(config.categories)}).\n"
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    default_target: str | None = None,
) -> int:
    default_prog = "jetski-token-viz" if default_target == "jetski" else None
    ns, cli_args = parse_cli_args(argv, prog=default_prog)
    prog_hint = (
        default_prog
        if default_prog is not None
        else (sys.argv[0] if argv is None else None)
    )

    try:
        config = VizConfig.from_env_and_args(
            target=ns.viz_target or default_target,
            prog_name=prog_hint,
            data_dir=ns.viz_data_dir,
            cli_args=cli_args,
            model=ns.viz_model,
            model_file=ns.viz_model_file,
            model_dir=ns.viz_model_dir,
            categories=ns.viz_categories,
            config_path=ns.viz_config,
            include_history=ns.viz_include_history,
            offline_heuristic_only=ns.viz_heuristic_only,
        )
    except ValueError as exc:
        err_tag = "jetski-token-viz" if (ns.viz_target or default_target) == "jetski" else "agy-token-viz"
        sys.stderr.write(f"[{err_tag}] Configuration error: {exc}\n")
        return 2

    tag = config.profile.tag

    if ns.viz_preload_only:
        try:
            return _run_preload(config)
        except Exception as exc:
            sys.stderr.write(f"[{tag}] Failed to preload model weights: {exc}\n")
            return 1

    explicit_bin = ns.viz_bin or (
        ns.viz_jetski_bin if config.target == "jetski" else ns.viz_agy_bin
    )
    if not explicit_bin:
        explicit_bin = ns.viz_agy_bin or ns.viz_jetski_bin

    try:
        real_cli = explicit_bin or discover_real_cli_binary(config.target)
    except FileNotFoundError as exc:
        sys.stderr.write(f"[{tag}] {exc}\n")
        return 127

    state = SessionTokenState(categories=config.categories)
    router = None if config.offline_heuristic_only else OpenModelRouter.from_config(config)
    classifier = AsyncStepClassifier(
        state=state,
        router=router,
        heuristic_only=config.offline_heuristic_only,
    )

    target_conv = extract_target_conversation_from_args(cli_args)
    is_resuming = is_resuming_conversation_from_args(cli_args)
    explicit_log = extract_log_file_from_args(cli_args)
    watcher = TranscriptWatcher(
        agy_data_dir=config.data_dir,
        log_subdirs=config.profile.log_subdirs,
        on_step=classifier.submit,
        target_conversation_id=target_conv,
        include_history=config.include_history,
        exclude_preexisting=not is_resuming,
        lock_to_first_conversation=True,
        on_baseline_context=lambda conv_id, prompt_tok, vis_end, end_ctx: (
            state.seed_conversation_baseline(
                conv_id,
                prev_prompt_tokens=prompt_tok,
                prev_visible_end_context=vis_end,
                prev_end_context=end_ctx,
            )
        ),
        on_reset=state.reset,
        explicit_log_file=explicit_log,
    )

    def _before_spawn() -> None:
        watcher.capture_baseline()

    def _after_spawn(child_pid: int) -> None:
        watcher.set_child_pid(child_pid)
        classifier.start()
        watcher.start()

    runner = PTYVisualizerRunner(
        cmd=[real_cli, *cli_args],
        state=state,
    )

    exit_code = runner.run(
        on_before_spawn=_before_spawn,
        on_process_spawn=_after_spawn,
    )

    # Flush final transcript lines and let pending step classifications finish
    watcher.stop(timeout=1.5)
    classifier.stop(timeout=3.0)

    use_color = should_use_color() and sys.stderr.isatty()
    summary = render_final_summary(state.snapshot(), use_color=use_color, label=tag)
    sys.stderr.write(f"{summary}\n")
    sys.stderr.flush()

    return exit_code


def main_jetski(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint for `jetski-token-viz`."""
    return main(argv, default_target="jetski")


if __name__ == "__main__":
    raise SystemExit(main())
