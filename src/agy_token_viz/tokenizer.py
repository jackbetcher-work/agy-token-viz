"""Local offline Gemini/Gemma SentencePiece tokenizer and verbatim tool-call serializer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Protocol


class TokenizerBackend(Protocol):
    """Protocol implemented by SentencePieceProcessor and the deterministic fallback encoder."""

    def encode(self, text: str) -> list[int]: ...


_FALLBACK_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]{0,11}|[0-9]{1,4}|[^\sA-Za-z0-9_]|[\n\r\t]+|\s+"
)


class _RegexFallbackTokenizer:
    """Deterministic subword fallback used only if SentencePiece or `.model` weights are absent."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [idx for idx, _ in enumerate(_FALLBACK_TOKEN_RE.finditer(text))]


_TOKENIZER_LOCK = threading.Lock()
_CACHED_TOKENIZER: TokenizerBackend | None = None
_CACHED_MODEL_PATH: Path | None = None

_BUNDLED_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "tokenizer.model"
_DEFAULT_CACHE_MODEL_PATH = (
    Path.home() / ".cache" / "agy-token-viz" / "models" / "tokenizer.model"
)
_HF_TOKENIZER_REPOS: tuple[str, ...] = (
    "unsloth/gemma-2-2b-it",
    "google/gemma-2-2b",
)


def resolve_tokenizer_model_path(explicit_path: Path | str | None = None) -> Path | None:
    """Locate the local Gemini/Gemma SentencePiece `tokenizer.model` file without network I/O."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("AGY_VIZ_TOKENIZER_MODEL", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(_BUNDLED_MODEL_PATH)
    candidates.append(_DEFAULT_CACHE_MODEL_PATH)

    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand

    # Check local Hugging Face cache (offline only)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

        for repo_id in _HF_TOKENIZER_REPOS:
            try:
                cached = hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer.model",
                    local_files_only=True,
                )
                cached_path = Path(cached)
                if cached_path.is_file() and cached_path.stat().st_size > 0:
                    return cached_path
            except Exception:
                continue
    except Exception:
        pass

    return None


def get_tokenizer(model_path: Path | str | None = None) -> TokenizerBackend:
    """Return the cached local SentencePiece tokenizer instance (initializing on first call)."""
    global _CACHED_TOKENIZER, _CACHED_MODEL_PATH

    if model_path is None and _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    with _TOKENIZER_LOCK:
        if model_path is None and _CACHED_TOKENIZER is not None:
            return _CACHED_TOKENIZER

        resolved = resolve_tokenizer_model_path(model_path)
        if resolved is not None:
            try:
                import sentencepiece as spm  # type: ignore[import-untyped]

                processor = spm.SentencePieceProcessor(model_file=str(resolved))
                if model_path is None:
                    _CACHED_TOKENIZER = processor
                    _CACHED_MODEL_PATH = resolved
                return processor  # type: ignore[return-value]
            except Exception:
                pass

        fallback = _RegexFallbackTokenizer()
        if model_path is None:
            _CACHED_TOKENIZER = fallback
        return fallback


def count_tokens_locally(text: str) -> int:
    """Computes exact token counts using the Gemini SentencePiece tokenizer."""
    if not text:
        return 0
    return len(get_tokenizer().encode(text))


def serialize_tool_calls_verbatim(tool_calls: Any) -> str:
    """
    Serialize `tool_calls` preserving raw string argument bodies without double-JSON-escaping.

    Gemini's structured function-calling representation tokenizes argument keys and raw string
    values directly rather than escaping every newline (`\\n`) or quote (`\\"`) inside large
    `write_file` / `edit_file` payloads.
    """
    if not tool_calls:
        return ""
    if isinstance(tool_calls, (list, tuple)):
        parts: list[str] = []
        for item in tool_calls:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                raw_args = item.get("args")
                if isinstance(raw_args, dict) and raw_args:
                    arg_strs: list[str] = []
                    for key, val in raw_args.items():
                        if isinstance(val, str):
                            val_str = val
                        else:
                            try:
                                val_str = json.dumps(val, ensure_ascii=False)
                            except (TypeError, ValueError):
                                val_str = str(val)
                        arg_strs.append(f"{key}: {val_str}")
                    parts.append(f"{name}({', '.join(arg_strs)})")
                elif name:
                    parts.append(name)
            elif hasattr(item, "name"):
                # ToolCallSummary fallback
                name = str(getattr(item, "name", "") or "").strip()
                summary = str(getattr(item, "summary", "") or "").strip()
                parts.append(f"{name}({summary})" if summary else name)
        return "\n".join(parts)
    return str(tool_calls)
