"""Configuration, harness profiles, model registry, and category definitions for agy-token-viz."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Literal, Mapping, Sequence


TargetCLI = Literal["agy", "jetski"]

HARNESS_OVERHEAD_CATEGORY = "Harness Overhead"

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "Planning",
    "Exploration",
    "Coding",
    "Commands",
    "Testing",
    "Communication",
)

# Short aliases for compact width-constrained terminal rendering
SHORT_CATEGORY_LABELS: dict[str, str] = {
    "Planning": "Plan",
    "Exploration": "Explore",
    "Coding": "Code",
    "Commands": "Cmd",
    "Testing": "Test",
    "Execution/Testing": "Exec",
    "Communication": "Comm",
    "Harness Overhead": "Harness",
    "System Overhead": "Sys",
    "System Setup": "Setup",
}


@dataclass(frozen=True)
class ModelSpec:
    """Describes a local or Hugging Face GGUF open-weight model."""

    name: str
    repo_id: str
    filename: str
    context_length: int = 2048
    local_path: Path | None = None

    def disk_target(self, model_dir: Path) -> Path:
        """Target path on disk where weights are stored and read from."""
        if self.local_path is not None:
            return self.local_path.expanduser().resolve()
        safe_repo = self.repo_id.strip("/").replace("/", "__")
        return model_dir.expanduser().resolve() / safe_repo / self.filename


DEFAULT_MODEL_PRESET = "qwen2.5-3b"

MODEL_PRESETS: dict[str, ModelSpec] = {
    "qwen2.5-3b": ModelSpec(
        name="qwen2.5-3b",
        repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        context_length=2048,
    ),
    "qwen2.5-7b": ModelSpec(
        name="qwen2.5-7b",
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        context_length=2048,
    ),
    "llama-3.2-3b": ModelSpec(
        name="llama-3.2-3b",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        context_length=2048,
    ),
    "llama-3.1-8b": ModelSpec(
        name="llama-3.1-8b",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        context_length=2048,
    ),
    "qwen2.5-1.5b": ModelSpec(
        name="qwen2.5-1.5b",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        context_length=2048,
    ),
    "llama-3.2-1b": ModelSpec(
        name="llama-3.2-1b",
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        context_length=2048,
    ),
    "qwen2.5-0.5b": ModelSpec(
        name="qwen2.5-0.5b",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        context_length=2048,
    ),
    "smollm2-360m": ModelSpec(
        name="smollm2-360m",
        repo_id="bartowski/SmolLM2-360M-Instruct-GGUF",
        filename="SmolLM2-360M-Instruct-Q4_K_M.gguf",
        context_length=2048,
    ),
}


def resolve_model_spec(
    model_identifier: str | None = None,
    filename_override: str | None = None,
) -> ModelSpec:
    """Resolve a preset name, local .gguf path, or '<org>/<repo>[:<file.gguf>]' string."""
    ident = (model_identifier or DEFAULT_MODEL_PRESET).strip()
    file_override = (filename_override or "").strip() or None

    preset_key = ident.lower()
    if preset_key in MODEL_PRESETS:
        preset = MODEL_PRESETS[preset_key]
        if file_override:
            return ModelSpec(
                name=preset.name,
                repo_id=preset.repo_id,
                filename=file_override,
                context_length=preset.context_length,
            )
        return preset

    # Check if pointing directly to a local .gguf file path
    candidate_path = Path(ident).expanduser()
    if ident.endswith(".gguf") and (":" not in ident) and (
        candidate_path.exists() or ident.startswith((".", "/", "~"))
    ):
        return ModelSpec(
            name=candidate_path.stem,
            repo_id="local",
            filename=candidate_path.name,
            local_path=candidate_path,
        )

    # Check '<org>/<repo>:<filename.gguf>' syntax
    if ":" in ident:
        repo_part, file_part = ident.split(":", 1)
        repo_part = repo_part.strip()
        file_part = (file_override or file_part).strip()
        if not repo_part or not file_part:
            raise ValueError(
                f"Invalid model spec '{ident}'. Expected '<org>/<repo>:<file.gguf>'."
            )
        short_name = repo_part.split("/")[-1]
        return ModelSpec(name=short_name, repo_id=repo_part, filename=file_part)

    # '<org>/<repo>' with separate filename override
    if "/" in ident and file_override:
        short_name = ident.split("/")[-1]
        return ModelSpec(name=short_name, repo_id=ident, filename=file_override)

    raise ValueError(
        f"Cannot resolve model '{ident}'. Provide a preset ({', '.join(MODEL_PRESETS)}), "
        "a local .gguf path, or '<org>/<repo>:<filename.gguf>' (or set AGY_VIZ_MODEL_FILE)."
    )


def parse_categories(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse and deduplicate activity categories while preserving order."""
    if raw is None:
        return DEFAULT_CATEGORIES
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    else:
        items = [str(part).strip() for part in raw]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            deduped.append(item)
    return tuple(deduped) if deduped else DEFAULT_CATEGORIES


def get_hf_token(env: Mapping[str, str] | None = None) -> str | None:
    """Retrieve Hugging Face authentication token from environment variables."""
    source = env if env is not None else os.environ
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = source.get(key, "").strip()
        if val:
            return val
    return None


@dataclass(frozen=True)
class HarnessProfile:
    """Target CLI harness metadata for binary, data directory, and log discovery."""

    name: TargetCLI
    display_name: str
    binary_name: str
    default_app_data_name: str
    data_dir_env: str
    data_dir_cfg_key: str
    real_bin_env: str
    log_subdirs: tuple[str, ...]
    fallback_bin_templates: tuple[str, ...]
    tag: str
    default_gemini_name: str = ".gemini"

    def default_gemini_dir(self) -> Path:
        return Path.home() / self.default_gemini_name

    def default_data_dir(self) -> Path:
        return self.default_gemini_dir() / self.default_app_data_name

    def candidate_fallback_bins(self) -> tuple[Path, ...]:
        return tuple(Path(p).expanduser() for p in self.fallback_bin_templates)


HARNESS_PROFILES: dict[TargetCLI, HarnessProfile] = {
    "agy": HarnessProfile(
        name="agy",
        display_name="Antigravity CLI (agy)",
        binary_name="agy",
        default_app_data_name="antigravity-cli",
        data_dir_env="AGY_DATA_DIR",
        data_dir_cfg_key="agy_data_dir",
        real_bin_env="AGY_REAL_BIN",
        log_subdirs=("log", "cli/log"),
        fallback_bin_templates=("~/.local/bin/agy",),
        tag="agy-token-viz",
    ),
    "jetski": HarnessProfile(
        name="jetski",
        display_name="Jetski CLI (jetski)",
        binary_name="jetski",
        default_app_data_name="jetski",
        data_dir_env="JETSKI_DATA_DIR",
        data_dir_cfg_key="jetski_data_dir",
        real_bin_env="JETSKI_REAL_BIN",
        log_subdirs=("cli/log", "log"),
        fallback_bin_templates=(
            "/usr/local/bin/jetski",
            "/usr/local/google/jetski-cli/jetski-cli",
            "~/.local/bin/jetski",
        ),
        tag="jetski-token-viz",
    ),
}


def resolve_harness_profile(
    target: str | None = None,
    prog_name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> HarnessProfile:
    """Resolve the active CLI harness profile (`agy` or `jetski`)."""
    source = env if env is not None else os.environ
    raw_target = (target or source.get("TOKEN_VIZ_TARGET", "")).strip().lower()
    if raw_target:
        if raw_target in HARNESS_PROFILES:
            return HARNESS_PROFILES[raw_target]  # type: ignore[index]
        raise ValueError(
            f"Unsupported target CLI '{raw_target}'. "
            f"Expected one of: {', '.join(HARNESS_PROFILES)}."
        )

    if prog_name:
        base = Path(prog_name).name.lower()
        if "jetski" in base:
            return HARNESS_PROFILES["jetski"]
        if "agy" in base:
            return HARNESS_PROFILES["agy"]

    return HARNESS_PROFILES["agy"]


def extract_data_dir_from_args(
    cli_args: Sequence[str],
    *,
    default_gemini_dir: Path | None = None,
    default_app_data_name: str = "antigravity-cli",
) -> Path | None:
    """
    Extract an overridden app data directory from forwarded CLI flags (`--gemini_dir`
    and/or `--app_data_dir`, supporting single/double dashes and `_`/`-` separators).
    """
    gemini_flags = {
        "--gemini_dir",
        "-gemini_dir",
        "--gemini-dir",
        "-gemini-dir",
    }
    app_data_flags = {
        "--app_data_dir",
        "-app_data_dir",
        "--app-data-dir",
        "-app-data-dir",
    }

    raw_gemini: str | None = None
    raw_app_data: str | None = None

    idx = 0
    n = len(cli_args)
    while idx < n:
        arg = cli_args[idx]
        if arg == "--":
            break
        if arg in gemini_flags and idx + 1 < n:
            val = cli_args[idx + 1].strip()
            if val:
                raw_gemini = val
            idx += 2
            continue
        if arg in app_data_flags and idx + 1 < n:
            val = cli_args[idx + 1].strip()
            if val:
                raw_app_data = val
            idx += 2
            continue

        matched_eq = False
        for prefix in gemini_flags:
            if arg.startswith(f"{prefix}="):
                val = arg.split("=", 1)[1].strip()
                if val:
                    raw_gemini = val
                matched_eq = True
                break
        if not matched_eq:
            for prefix in app_data_flags:
                if arg.startswith(f"{prefix}="):
                    val = arg.split("=", 1)[1].strip()
                    if val:
                        raw_app_data = val
                    matched_eq = True
                    break
        idx += 1

    if raw_gemini is None and raw_app_data is None:
        return None

    if raw_gemini is not None:
        gemini_path = Path(raw_gemini).expanduser()
        if not gemini_path.is_absolute():
            gemini_path = Path.home() / gemini_path
    else:
        gemini_path = (
            default_gemini_dir
            if default_gemini_dir is not None
            else (Path.home() / ".gemini")
        )

    app_name = raw_app_data if raw_app_data is not None else default_app_data_name
    app_path = Path(app_name).expanduser()
    if app_path.is_absolute():
        return app_path
    return gemini_path / app_path


def discover_real_cli_binary(
    target: str = "agy",
    wrapper_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Locate the underlying target CLI executable (`agy` or `jetski`) without matching our shim."""
    source = env if env is not None else os.environ
    profile = resolve_harness_profile(target=target, env=source)

    for env_key in (profile.real_bin_env, "TOKEN_VIZ_REAL_BIN"):
        explicit = source.get(env_key, "").strip()
        if explicit:
            return explicit

    raw_path = source.get("PATH", "")
    excluded_dirs: set[Path] = set()
    if wrapper_dir is not None:
        try:
            excluded_dirs.add(wrapper_dir.resolve())
        except OSError:
            pass

    project_bin = Path(__file__).resolve().parents[2] / "bin"
    shim_realpath: Path | None = None
    try:
        excluded_dirs.add(project_bin.resolve())
        shim_candidate = project_bin / profile.binary_name
        if shim_candidate.exists():
            shim_realpath = shim_candidate.resolve()
    except OSError:
        pass

    def _is_our_shim(candidate: Path) -> bool:
        if shim_realpath is None:
            return False
        try:
            return candidate.resolve() == shim_realpath
        except OSError:
            return False

    for entry in raw_path.split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).resolve() in excluded_dirs:
                continue
        except OSError:
            pass
        found = shutil.which(profile.binary_name, path=entry)
        if found and not _is_our_shim(Path(found)):
            return found

    for fallback in profile.candidate_fallback_bins():
        if fallback.is_file() and os.access(fallback, os.X_OK) and not _is_our_shim(fallback):
            return str(fallback)

    raise FileNotFoundError(
        f"Could not locate the real '{profile.binary_name}' CLI binary on PATH. "
        f"Ensure {profile.display_name} is installed or set {profile.real_bin_env}=/path/to/{profile.binary_name}."
    )


def discover_real_agy_binary(
    wrapper_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Locate the underlying `agy` executable without matching our own wrapper shim."""
    return discover_real_cli_binary("agy", wrapper_dir=wrapper_dir, env=env)


def _first_env(source: Mapping[str, str], keys: Sequence[str]) -> str | None:
    for key in keys:
        val = source.get(key, "").strip()
        if val:
            return val
    return None


@dataclass(frozen=True)
class VizConfig:
    """Runtime configuration for agy-token-viz / jetski-token-viz."""

    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    model_spec: ModelSpec = field(
        default_factory=lambda: MODEL_PRESETS[DEFAULT_MODEL_PRESET]
    )
    model_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "agy-token-viz" / "models"
    )
    agy_data_dir: Path = field(
        default_factory=lambda: Path.home() / ".gemini" / "antigravity-cli"
    )
    hf_token: str | None = None
    real_agy_bin: str = "agy"
    n_gpu_layers: int = -1
    n_threads: int = 4
    include_history: bool = False
    offline_heuristic_only: bool = False
    target: TargetCLI = "agy"

    @property
    def profile(self) -> HarnessProfile:
        return resolve_harness_profile(self.target)

    @property
    def data_dir(self) -> Path:
        return self.agy_data_dir

    @property
    def real_cli_bin(self) -> str:
        return self.real_agy_bin

    @classmethod
    def from_env_and_args(
        cls,
        *,
        target: str | None = None,
        prog_name: str | None = None,
        data_dir: str | Path | None = None,
        cli_args: Sequence[str] | None = None,
        model: str | None = None,
        model_file: str | None = None,
        model_dir: str | Path | None = None,
        categories: str | Sequence[str] | None = None,
        config_path: str | Path | None = None,
        include_history: bool = False,
        offline_heuristic_only: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> "VizConfig":
        source = env if env is not None else os.environ
        profile = resolve_harness_profile(target=target, prog_name=prog_name, env=source)
        target_prefix = "JETSKI_VIZ" if profile.name == "jetski" else "AGY_VIZ"

        def _viz_env(suffix: str) -> str | None:
            keys: list[str] = [f"{target_prefix}_{suffix}", f"TOKEN_VIZ_{suffix}"]
            if "AGY_VIZ_" + suffix not in keys:
                keys.append(f"AGY_VIZ_{suffix}")
            return _first_env(source, keys)

        # Optional JSON config file
        file_cfg: dict[str, object] = {}
        cfg_file_str = (
            str(config_path)
            if config_path is not None
            else (_viz_env("CONFIG") or "")
        )
        default_cfg_path = Path.home() / ".config" / "agy-token-viz" / "config.json"
        target_cfg_path = Path(cfg_file_str).expanduser() if cfg_file_str else default_cfg_path
        if target_cfg_path.is_file():
            try:
                loaded = json.loads(target_cfg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    file_cfg = loaded
            except (OSError, json.JSONDecodeError):
                pass

        resolved_categories = parse_categories(
            categories
            or _viz_env("CATEGORIES")
            or (
                file_cfg.get("categories")
                if isinstance(file_cfg.get("categories"), (str, list))
                else None
            )
        )

        raw_model = (
            model
            or _viz_env("MODEL")
            or (str(file_cfg["model"]) if "model" in file_cfg else None)
            or DEFAULT_MODEL_PRESET
        )
        raw_model_file = (
            model_file
            or _viz_env("MODEL_FILE")
            or (str(file_cfg["model_file"]) if "model_file" in file_cfg else None)
        )
        spec = resolve_model_spec(raw_model, raw_model_file)

        raw_model_dir = (
            model_dir
            or _viz_env("MODEL_DIR")
            or (str(file_cfg["model_dir"]) if "model_dir" in file_cfg else None)
            or (Path.home() / ".cache" / "agy-token-viz" / "models")
        )
        resolved_model_dir = Path(raw_model_dir).expanduser()

        arg_data_dir = (
            extract_data_dir_from_args(
                cli_args,
                default_gemini_dir=profile.default_gemini_dir(),
                default_app_data_name=profile.default_app_data_name,
            )
            if cli_args
            else None
        )
        cfg_data_dir = (
            str(file_cfg[profile.data_dir_cfg_key])
            if profile.data_dir_cfg_key in file_cfg
            else (str(file_cfg["data_dir"]) if "data_dir" in file_cfg else None)
        )
        raw_data_dir = (
            data_dir
            or arg_data_dir
            or _first_env(source, ("TOKEN_VIZ_DATA_DIR", profile.data_dir_env))
            or cfg_data_dir
            or profile.default_data_dir()
        )
        resolved_data_dir = Path(raw_data_dir).expanduser()

        raw_gpu_layers = _viz_env("GPU_LAYERS")
        n_gpu_layers = int(
            raw_gpu_layers
            if raw_gpu_layers is not None
            else file_cfg.get("n_gpu_layers", -1)
        )
        raw_threads = _viz_env("THREADS")
        n_threads = int(
            raw_threads
            if raw_threads is not None
            else file_cfg.get("n_threads", 4)
        )
        raw_heuristic = (_viz_env("HEURISTIC_ONLY") or "").lower()
        heuristic_only = offline_heuristic_only or raw_heuristic in ("1", "true", "yes")

        try:
            real_bin = discover_real_cli_binary(profile.name, env=source)
        except FileNotFoundError:
            real_bin = profile.binary_name

        return cls(
            categories=resolved_categories,
            model_spec=spec,
            model_dir=resolved_model_dir,
            agy_data_dir=resolved_data_dir,
            hf_token=get_hf_token(source),
            real_agy_bin=real_bin,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            include_history=include_history,
            offline_heuristic_only=heuristic_only,
            target=profile.name,
        )
