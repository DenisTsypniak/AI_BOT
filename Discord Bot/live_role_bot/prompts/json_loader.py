from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("live_role_bot.prompts")

_CACHE: dict[str, tuple[int | None, dict[str, Any]]] = {}


def _data_dir() -> Path:
    return Path(__file__).with_name("data")


def _read_text(path: Path) -> str:
    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unable to read prompt JSON: {path}")


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_prompt_json(filename: str, defaults: dict[str, Any]) -> dict[str, Any]:
    data_dir = _data_dir()
    path = data_dir / filename
    cache_key = str(path.resolve())

    mtime_ns: int | None = None
    if path.exists():
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None

    cached = _CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime_ns:
        return copy.deepcopy(cached[1])

    merged_defaults = copy.deepcopy(defaults)
    if not path.exists():
        logger.warning("Prompt JSON not found: %s (using defaults)", path)
        _CACHE[cache_key] = (mtime_ns, merged_defaults)
        return copy.deepcopy(merged_defaults)

    try:
        payload = json.loads(_read_text(path))
    except Exception as exc:
        logger.warning("Failed to parse prompt JSON %s (%s). Using defaults.", path, exc)
        _CACHE[cache_key] = (mtime_ns, merged_defaults)
        return copy.deepcopy(merged_defaults)

    if not isinstance(payload, dict):
        logger.warning("Prompt JSON root must be an object: %s (using defaults)", path)
        _CACHE[cache_key] = (mtime_ns, merged_defaults)
        return copy.deepcopy(merged_defaults)

    merged = _deep_merge(merged_defaults, payload)
    if not isinstance(merged, dict):
        merged = merged_defaults

    _CACHE[cache_key] = (mtime_ns, copy.deepcopy(merged))
    return merged

