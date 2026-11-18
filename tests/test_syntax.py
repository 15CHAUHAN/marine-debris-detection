"""Guard: every Python file in the repo must at least parse.

A one-character typo in scripts/download_merge_drone_dataset.py ("split_n ame")
once made a whole script unrunnable without any test noticing, because no test
imports the scripts. Parsing every file is cheap and needs no data.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("src", "scripts", "dashboard", "tests")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in PACKAGES:
        files.extend(
            p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in p.parts
        )
    return sorted(files)


def test_python_files_were_discovered():
    assert len(_python_files()) > 50


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_python_file_parses(path: Path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
