"""Shared utilities: file loading, JSON extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .types import PROJECT_ROOT


def load_prompt(name: str) -> str:
    """Read a prompt file from prompts/{name}.md."""
    path = PROJECT_ROOT / "prompts" / f"{name}.md"
    return path.read_text()


def load_memory() -> str:
    """Read memory.md, returning empty string if missing."""
    try:
        return (PROJECT_ROOT / "memory.md").read_text()
    except FileNotFoundError:
        return ""


def load_program() -> str:
    """Read program.md, returning empty string if missing."""
    try:
        return (PROJECT_ROOT / "program.md").read_text()
    except FileNotFoundError:
        return ""


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts. Returns [] if file missing."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def extract_json(text: str) -> dict:
    """Parse JSON from text, falling back to regex extraction of first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise
