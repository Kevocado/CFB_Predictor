#!/usr/bin/env python3
"""Mutation verification for the CFBD request timeout (tests/test_team_stats.py).

Each mutant is an exact-string source replacement that breaks exactly one
stated property of the timeout fix. The replacement is asserted to have applied
before the tests run, so a mutant that silently changed nothing is reported as
VACUOUS rather than as a survivor.

A mutant that changes the file and leaves the suite green is a HOLE in the test.

Nothing here reaches the network. tests/test_team_stats.py stubs the `cfbd`
module in `sys.modules` (`fake_cfbd`), and tests/test_startup_seed.py carries
a tripwire that fails loudly if any mutant manages to call `fetch_team_stats`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "cfb_predictor" / "data" / "team_stats.py"
TARGETS = "tests/test_team_stats.py tests/test_startup_seed.py"
PY = "/Users/sigey/Documents/Projects.nosync/CFB_Predictor/.venv/bin/python"


def tests() -> tuple[str, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", *TARGETS.split(), "-q", "-m", "not network", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return proc.stdout.strip().splitlines()[-1], proc.stdout


MUTANTS: list[tuple[str, str, str]] = [
    # (label, old, new)
    (
        "the timeout is not passed to the client (wait forever again)",
        "                year=season, week=week, _request_timeout=CFBD_REQUEST_TIMEOUT\n",
        "                year=season, week=week\n",
    ),
    (
        "the bound is None (urllib3 reads that as no timeout)",
        "CFBD_REQUEST_TIMEOUT = 60",
        "CFBD_REQUEST_TIMEOUT = None",
    ),
    (
        "the bound is zero (non-positive means no bound)",
        "CFBD_REQUEST_TIMEOUT = 60",
        "CFBD_REQUEST_TIMEOUT = 0",
    ),
    (
        "the bound is unbounded in practice (one stall outlives boot patience)",
        "CFBD_REQUEST_TIMEOUT = 60",
        "CFBD_REQUEST_TIMEOUT = 3600",
    ),
    (
        "a timed-out week is swallowed and tombstoned (silent stall, then a lie)",
        "            raw = api.get_game_team_stats(\n"
        "                year=season, week=week, _request_timeout=CFBD_REQUEST_TIMEOUT\n"
        "            )",
        "            try:\n"
        "                raw = api.get_game_team_stats(\n"
        "                    year=season, week=week, _request_timeout=CFBD_REQUEST_TIMEOUT\n"
        "                )\n"
        "            except TimeoutError:\n"
        "                frame = _empty_week_frame(season, week)\n"
        "                frame.to_parquet(path, index=False)\n"
        "                continue",
    ),
    (
        "a timed-out week is retried without bound awareness (a metered multiplier)",
        "            raw = api.get_game_team_stats(\n"
        "                year=season, week=week, _request_timeout=CFBD_REQUEST_TIMEOUT\n"
        "            )",
        "            for _attempt in range(3):\n"
        "                try:\n"
        "                    raw = api.get_game_team_stats(\n"
        "                        year=season, week=week, _request_timeout=CFBD_REQUEST_TIMEOUT\n"
        "                    )\n"
        "                    break\n"
        "                except TimeoutError:\n"
        "                    continue",
    ),
]


def main() -> int:
    original = SRC.read_text()

    def restore() -> None:
        SRC.write_text(original)

    baseline, _ = tests()
    print(f"baseline (unmutated): {baseline}\n")
    if " failed" in baseline or "error" in baseline:
        print("BASELINE IS NOT GREEN -- fix that before trusting any mutant result.")
        return 1

    survivors = []
    vacuous = []
    for label, old, new in MUTANTS:
        count = original.count(old)
        if count != 1:
            vacuous.append((label, f"anchor appears {count}x, expected 1"))
            print(f"VACUOUS    {label}  (anchor appears {count}x -- not a mutation)")
            continue
        SRC.write_text(original.replace(old, new, 1))
        last, out = tests()
        caught = " failed" in last or "error" in last
        who = sorted(
            {
                line.split(" ")[1].split("::")[1]
                for line in out.splitlines()
                if line.startswith("FAILED ")
            }
        )
        if caught:
            print(f"CAUGHT     {label}")
            print(f"           by {', '.join(w.split('[')[0] for w in who)}")
        else:
            print(f"SURVIVED   {label}   <-- HOLE IN THE TEST")
            survivors.append(label)
        restore()

    print("\nrestored; confirming clean:")
    print(f"  {tests()[0]}")
    assert SRC.read_text() == original, f"{SRC} not restored"

    print(f"\nmutants: {len(MUTANTS)}  caught: {len(MUTANTS) - len(survivors) - len(vacuous)}"
          f"  survived: {len(survivors)}  vacuous: {len(vacuous)}")
    for label in survivors:
        print(f"  SURVIVED: {label}")
    for label, why in vacuous:
        print(f"  VACUOUS:  {label} ({why})")
    return 1 if survivors or vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
