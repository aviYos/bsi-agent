"""Shared data utilities."""

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Load data from JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path):
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, default=str) + "\n")


def load_config(path: Path) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
