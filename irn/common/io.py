"""Atomic file I/O utilities enforcing Constraint C-7."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Union


def atomic_write_text(filepath: Union[str, Path], content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text to filepath via a temporary sibling file."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding=encoding) as tmp:
        tmp.write(content)
        tmp_name = tmp.name

    os.replace(tmp_name, path)
    return path


def atomic_write_json(filepath: Union[str, Path], data: Any, indent: int = 2) -> Path:
    """Atomically write Python object to JSON file."""
    content = json.dumps(data, indent=indent, default=str)
    return atomic_write_text(filepath, content)


def atomic_write_bytes(filepath: Union[str, Path], data: bytes) -> Path:
    """Atomically write raw bytes to filepath."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_name = tmp.name

    os.replace(tmp_name, path)
    return path
