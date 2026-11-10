"""Configuration, model registry, and category definitions for agy-token-viz."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence


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


def discover_real_agy_binary(
    wrapper_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Locate the underlying `agy` executable without matching our own wrapper shim."""
    source = env if env is not None else os.environ
    explicit = source.get("AGY_REAL_BIN", "").strip()
    if explicit:
        return explicit

    raw_path = source.get("PATH", "")
    excluded: set[Path] = set()
    if wrapper_dir is not None:
        try:
            excluded.add(wrapper_dir.resolve())
        except OSError:
            pass
    # Also exclude the project's own bin/ folder
    project_bin = Path(__file__).resolve().parents[2] / "bin"
    try:
        excluded.add(project_bin.resolve())
    except OSError:
        pass

    filtered_parts: list[str] = []
    for entry in raw_path.split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).resolve() in excluded:
                continue
        except OSError:
            pass
        filtered_parts.append(entry)

    filtered_path = os.pathsep.join(filtered_parts)
    found = shutil.which("agy", path=filtered_path)
    if found:
        return found

    default_local = Path.home() / ".local" / "bin" / "agy"
    if default_local.is_file() and os.access(default_local, os.X_OK):
        return str(default_local)

    raise FileNotFoundError(
        "Could not locate the real 'agy' CLI binary on PATH. "
        "Ensure Antigravity CLI is installed or set AGY_REAL_BIN=/path/to/agy."
    )


@dataclass(frozen=True)
class VizConfig:
    """Runtime configuration for agy-token-viz."""

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

    @classmethod
    def from_env_and_args(
        cls,
        *,
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

        # Optional JSON config file
        file_cfg: dict[str, object] = {}
        cfg_file_str = (
            str(config_path)
            if config_path is not None
            else source.get("AGY_VIZ_CONFIG", "").strip()
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
            or source.get("AGY_VIZ_CATEGORIES")
            or (file_cfg.get("categories") if isinstance(file_cfg.get("categories"), (str, list)) else None)
        )

        raw_model = (
            model
            or source.get("AGY_VIZ_MODEL")
            or (str(file_cfg["model"]) if "model" in file_cfg else None)
            or DEFAULT_MODEL_PRESET
        )
        raw_model_file = (
            model_file
            or source.get("AGY_VIZ_MODEL_FILE")
            or (str(file_cfg["model_file"]) if "model_file" in file_cfg else None)
        )
        spec = resolve_model_spec(raw_model, raw_model_file)

        raw_model_dir = (
            model_dir
            or source.get("AGY_VIZ_MODEL_DIR")
            or (str(file_cfg["model_dir"]) if "model_dir" in file_cfg else None)
            or (Path.home() / ".cache" / "agy-token-viz" / "models")
        )
        resolved_model_dir = Path(raw_model_dir).expanduser()

        raw_agy_data = (
            source.get("AGY_DATA_DIR")
            or (str(file_cfg["agy_data_dir"]) if "agy_data_dir" in file_cfg else None)
            or (Path.home() / ".gemini" / "antigravity-cli")
        )
        resolved_agy_data = Path(raw_agy_data).expanduser()

        n_gpu_layers = int(source.get("AGY_VIZ_GPU_LAYERS", file_cfg.get("n_gpu_layers", -1)))
        n_threads = int(source.get("AGY_VIZ_THREADS", file_cfg.get("n_threads", 4)))
        heuristic_only = offline_heuristic_only or source.get(
            "AGY_VIZ_HEURISTIC_ONLY", ""
        ).lower() in ("1", "true", "yes")

        try:
            real_bin = discover_real_agy_binary(env=source)
        except FileNotFoundError:
            real_bin = "agy"

        return cls(
            categories=resolved_categories,
            model_spec=spec,
            model_dir=resolved_model_dir,
            agy_data_dir=resolved_agy_data,
            hf_token=get_hf_token(source),
            real_agy_bin=real_bin,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            include_history=include_history,
            offline_heuristic_only=heuristic_only,
        )
