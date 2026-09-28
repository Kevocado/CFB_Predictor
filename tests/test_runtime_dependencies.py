"""Every third-party module imported under src/ must be a runtime dependency.

This caught a production outage. `api/explain.py` imports `httpx` at runtime,
but F1 did not declare it at all and NBA listed it only under the `dev` extra,
so `docker build && docker run` produced an image that died on import with
`ModuleNotFoundError: No module named 'httpx'`. Both services were rolled back
from the VPS.

The test suite could not see it: httpx arrives in the venv through dev/test
tooling (fastapi's own test client, respx, pytest tooling), so every test passed
and the image was broken. That is the whole point of this test — it reads
pyproject.toml, not the venv, so it asks the question the venv cannot answer.

A distribution can satisfy more than one module name, so there is a small
explicit map. Keep it short and keep it honest: a new import with no entry here
fails the test, which is the intended pressure.
"""
from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# Top-level module name -> import name on PyPI, where they differ.
MODULE_TO_DISTRIBUTION = {
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    # Vendored in-tree, or a stdlib module we must not look for.
    "pl_predictor": None,
    "nfl_predictor": None,
    "cfb_predictor": None,
}

STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", ()))


def _declared() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    runtime = data["project"]["dependencies"]
    return {re.split(r"[<>=!\[ ]", dep.strip())[0].lower().replace("_", "-") for dep in runtime}


def _imported_modules() -> set[str]:
    """Every top-level module imported by first-party code under src/."""
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import: in-tree by definition
                    continue
                if node.module:
                    found.add(node.module.split(".")[0])
    return found


def test_the_sweep_finds_the_imports_it_is_guarding():
    """Guards the sweep below against passing because it found nothing.

    Every repo here imports fastapi and pydantic, so if those disappear from
    src/ this file is wrong and should be updated deliberately rather than
    quietly asserting nothing.
    """
    found = _imported_modules()
    assert {"fastapi", "pydantic"} <= found, (
        f"the sweep only sees {sorted(found)}; update this test deliberately"
    )


@pytest.mark.parametrize("module", sorted(_imported_modules()))
def test_import_is_a_runtime_dependency(module: str):
    if module in STDLIB:
        pytest.skip(f"{module} is stdlib")
    if module.startswith("_"):
        pytest.skip("private module")

    if module in MODULE_TO_DISTRIBUTION:
        distribution = MODULE_TO_DISTRIBUTION[module]
        if distribution is None:
            pytest.skip(f"{module} is first-party")
    else:
        distribution = module.lower().replace("_", "-")

    declared = _declared()
    assert distribution in declared or module.lower().replace("_", "-") in declared, (
        f"{module} is imported under src/ but {distribution!r} is not in "
        f"[project].dependencies. A Docker image installs only the runtime "
        f"dependencies, so this crashes at import on the VPS while every test "
        f"passes (test tooling pulls it in). Add it there; a dev extra will not do."
    )
