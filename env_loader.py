"""
env_loader.py - Lightweight .env discovery for local CLI runs.

This project is usually run as a script (`python main.py`) rather than through a
framework that auto-loads environment variables. To keep the CLI ergonomic
without adding a dependency on python-dotenv, we load a nearby `.env` file if
Anthropic credentials are not already present in the process environment.
"""

from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _PROJECT_ROOT.parent


def _parse_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                data[key] = value
    except OSError:
        return {}
    return data


def _candidate_paths() -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    # Standard local project locations first.
    for base in (Path.cwd(), _PROJECT_ROOT):
        current = base.resolve()
        while True:
            _add(current / ".env")
            if current == current.parent or current == _WORKSPACE_ROOT:
                break
            current = current.parent

    # Then look for common sibling-project layouts within the workspace.
    try:
        for child in _WORKSPACE_ROOT.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            _add(child / ".env")
            _add(child / "backend" / ".env")
    except OSError:
        pass

    return candidates


def load_local_env() -> Path | None:
    """
    Load Anthropic-related settings from a nearby .env file if missing.

    Returns the path used, or None if nothing was loaded.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return None

    for path in _candidate_paths():
        if not path.is_file():
            continue
        parsed = _parse_dotenv(path)
        api_key = parsed.get("ANTHROPIC_API_KEY")
        if not api_key:
            continue

        os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        if parsed.get("CLAUDE_MODEL"):
            os.environ.setdefault("CLAUDE_MODEL", parsed["CLAUDE_MODEL"])
        return path

    return None
