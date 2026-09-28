#!/usr/bin/env python3
"""Mutation verification for tests/test_startup_seed.py.

Each mutant is an exact-string source replacement that breaks exactly one stated
property. The replacement is asserted to have applied before the tests run, so a
mutant that silently changed nothing is reported as VACUOUS rather than as a
survivor -- the two are completely different findings and conflating them is how
a mutation harness ends up being trusted when it is not.

A mutant that changes the file and leaves the suite green is a HOLE in the test.

Nothing here reaches the network. tests/test_startup_seed.py carries a tripwire
that fails loudly if any mutant manages to call `fetch_team_stats`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "cfb_predictor" / "startup_seed.py"
DOCKERFILE = REPO / "Dockerfile"
TARGET = "tests/test_startup_seed.py"
PY = "/Users/sigey/Documents/Projects.nosync/CFB_Predictor/.venv/bin/python"

KEY_LINE = '    if not str(env.get("CFBD_API_KEY", "")).strip():\n'


def tests() -> tuple[str, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", TARGET, "-q", "-m", "not network", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    return proc.stdout.strip().splitlines()[-1], proc.stdout


MUTANTS: list[tuple[str, str, str | None, str]] = [
    # (label, file, old, new) -- file "src" is startup_seed.py, "dockerfile" is Dockerfile
    (
        "re-seeds a populated cache",
        "src",
        "    if state is CacheState.COMPLETE:",
        "    if False:",
    ),
    (
        "no COMPLETE short-circuit",
        "src",
        "    elif cached == total:",
        "    elif False:",
    ),
    (
        "populated cache classified EMPTY",
        "src",
        "    if cached == 0 and unreadable == 0:",
        "    if cached == total:",
    ),
    (
        "partial cache re-billed with no flag",
        "src",
        "    if state is CacheState.PARTIAL and not config.resume_partial:",
        "    if False:",
    ),
    (
        "torn files invisible -> cache looks empty",
        "src",
        "    if cached == 0 and unreadable == 0:",
        "    if cached == 0:",
    ),
    (
        "seeds with no CFBD_API_KEY (plan)",
        "src",
        KEY_LINE + "        return _plan(\n            Action.SKIP,",
        "        return _plan(\n            Action.SEED,",
    ),
    (
        "executor key re-check removed",
        "src",
        '    if not str(env.get("CFBD_API_KEY", "")).strip():\n'
        "        raise RuntimeError(",
        "    if False:\n        raise RuntimeError(",
    ),
    (
        "bad scope coerced to the default -> a bill",
        "src",
        "    except ValueError as exc:\n        return SeedConfig(",
        "    except ValueError as exc:\n        return config_from_env({}) or SeedConfig(",
    ),
    (
        "reversed from/to-year accepted",
        "src",
        "    if from_year > to_year:",
        "    if False:",
    ),
    (
        "boot task propagates the failure (crashloop)",
        "src",
        "    except Exception as exc:  # noqa: BLE001 - a boot task must not kill the process",
        "    except BaseException as _never:",
    ),
    (
        "retries the backfill once (two bills)",
        "src",
        "        executor(plan)\n",
        "        executor(plan)\n        executor(plan)\n",
    ),
    (
        "entrypoint exits non-zero on failure",
        "src",
        "    print(result.status_line, flush=True)",
        "    print(result.status_line, flush=True)\n"
        "    return 0 if result.outcome is not Outcome.FAILED else 1",
    ),
    (
        "a mistyped bool falls back to the default",
        "src",
        "    raise ValueError(f\"{name} must be 'true' or 'false', got {raw!r}\")",
        "    return default",
    ),
    (
        "out-of-range week accepted (a billed empty call)",
        "src",
        "    if out_of_range:",
        "    if False:",
    ),
    (
        "seeder removed from the container boot path",
        "dockerfile",
        "python -m cfb_predictor.startup_seed; ",
        "",
    ),
    (
        "seeder chained with && (a crashloop)",
        "dockerfile",
        "startup_seed; exec uvicorn",
        "startup_seed && exec uvicorn",
    ),
    (
        "server not exec'd (no signal delivery)",
        "dockerfile",
        "; exec uvicorn",
        "; uvicorn",
    ),
]


def main() -> int:
    originals = {SRC: SRC.read_text(), DOCKERFILE: DOCKERFILE.read_text()}

    def restore() -> None:
        for path, text in originals.items():
            path.write_text(text)

    baseline, _ = tests()
    print(f"baseline (unmutated): {baseline}\n")
    if " failed" in baseline or "error" in baseline:
        print("BASELINE IS NOT GREEN -- fix that before trusting any mutant result.")
        return 1

    survivors = []
    vacuous = []
    for label, which, old, new in MUTANTS:
        target = SRC if which == "src" else DOCKERFILE
        text = originals[target]
        count = text.count(old)
        if count != 1:
            vacuous.append((label, f"anchor appears {count}x, expected 1"))
            print(f"VACUOUS    {label}  (anchor appears {count}x -- not a mutation)")
            continue
        target.write_text(text.replace(old, new, 1))
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
    for path in originals:
        assert path.read_text() == originals[path], f"{path} not restored"

    print(f"\nmutants: {len(MUTANTS)}  caught: {len(MUTANTS) - len(survivors) - len(vacuous)}"
          f"  survived: {len(survivors)}  vacuous: {len(vacuous)}")
    for label in survivors:
        print(f"  SURVIVED: {label}")
    for label, why in vacuous:
        print(f"  VACUOUS:  {label} ({why})")
    return 1 if survivors or vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
