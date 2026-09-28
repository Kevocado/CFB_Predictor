"""startup_seed.py -- decide at boot whether to seed the team-stats cache, and say so.

The problem this exists to solve is a **misleading assumption**, not an outage.
`data/team_stats.py` documents a 2004-2025 reconciliation measured over 42,190
team-games. That measurement is real. The data behind it is not in production:
`data/cache/` is gitignored *and* listed in `.dockerignore`, and the Dockerfile
`mkdir`s the cache directories empty, so a fresh container has no team-stats
cache and no way to acquire one. A reader of that docstring has no way to tell
the difference.

Nothing breaks, because `reconcile_against_players` has no serving caller. So
this module does not add a route, a schedule, or a retry loop -- it makes the
state **legible and self-healing** at the only moment a human or a machine can
act on it, which is container start.

**Everything here is money.** CFBD is metered, one call per (season, week), and
the full documented scope is 22 seasons x 16 weeks = **352 calls in a single
boot**. So the rules are conservative and the defaults lean towards *not*
spending:

- **Idempotent, by construction.** The decision is made by reading the cache the
  fetcher itself would read, with the fetcher's own `_read_cache` test, so "the
  seeder thinks it is done" and "the fetcher would re-request this" cannot
  disagree. A populated cache is a fast no-op and reaches no network code.
- **Never a loop.** One attempt per process, wrapped in a bare `except
  Exception`, and the entrypoint always exits 0. A crashloop is the worst
  outcome available here: the platform restarts the container, the boot task
  runs again, and the bill is attempted again on every restart.
- **No key, no attempt.** Not "attempt and hope" -- blocked, named, zero calls.
- **A nonsense scope is declined, not guessed.** Correcting an operator's typo
  into a bill at 3am is worse than not seeding today.
- **A partial or unreadable cache is not re-billed unattended.** Those files
  were already paid for once. They are reported as partial so the log says
  "broken", not "never fetched"; `CFB_SEED_RESUME_PARTIAL=true` is the lever.

**An empty cache and a failed one are different states and the log says which.**
Both render `cache=empty` (it is), but the outcome differs -- `outcome=blocked`
with a named reason versus `outcome=failed` with the error. An operator reading
one line at 3am needs to know whether to go and set a key or to go and look at
CFBD.

Configuration (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `CFB_SEED_TEAM_STATS` | `true` | master switch |
| `CFB_SEED_FROM_YEAR` | `2004` | first season |
| `CFB_SEED_TO_YEAR` | `2025` | last season |
| `CFB_SEED_WEEKS` | `1-16` | `1-16`, or a list like `1,2,3` |
| `CFB_SEED_RESUME_PARTIAL` | `false` | seed a partly-populated cache |
| `CFBD_API_KEY` | -- | required before any call is attempted |

The default is the full 2004-2025 sweep because that is the population
`data/team_stats.py`'s measurements are stated over; narrowing it silently
would leave the docstring and the cache describing different things, which is
the exact confusion this module is here to remove. If 352 calls is not
affordable, set the scope rather than accepting a partial cache silently.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_FROM_YEAR = 2004
DEFAULT_TO_YEAR = 2025
DEFAULT_WEEKS = tuple(range(1, 17))

# CFBD's own parameter range for `get_game_team_stats(week=)`. A week outside
# this is a billed call that returns nothing -- the same money bug
# scripts/backfill_team_stats.py already refuses, and refused for the same reason.
MIN_WEEK, MAX_WEEK = 1, 16


class Action(str, Enum):
    SEED = "seed"
    SKIP = "skip"


class CacheState(str, Enum):
    EMPTY = "empty"
    PARTIAL = "partial"
    COMPLETE = "complete"


class Outcome(str, Enum):
    SEEDED = "seeded"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class SeedConfig:
    """What the operator asked for, and whether it was coherent."""

    enabled: bool
    from_year: int
    to_year: int
    weeks: tuple[int, ...]
    resume_partial: bool
    error: str | None = None

    @property
    def seasons(self) -> list[int]:
        return list(range(self.from_year, self.to_year + 1))

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [(season, week) for season in self.seasons for week in self.weeks]


@dataclass(frozen=True)
class SeedPlan:
    """The decision, with the evidence it was made from.

    `cached_pairs` counts files the fetcher would accept. `unreadable_pairs`
    counts files that exist but would not -- a torn write or a schema change.
    They are separated because they call for different responses: the first
    means the work is done, the second means it was paid for and then lost.

    `outcome` is what the run *will* be if nothing is fetched: `SKIPPED` when
    there is genuinely nothing to do, `BLOCKED` when the cache is unusable and
    something outside the container has to change first. It is decided here
    rather than in the runner so that the distinction is inspectable without
    executing anything.
    """

    action: Action
    state: CacheState
    reason: str
    config: SeedConfig
    cached_pairs: int
    unreadable_pairs: int
    total_pairs: int
    outcome: Outcome

    @property
    def should_seed(self) -> bool:
        return self.action is Action.SEED

    @property
    def missing_pairs(self) -> int:
        return self.total_pairs - self.cached_pairs

    @property
    def estimated_calls(self) -> int:
        return self.missing_pairs if self.should_seed else 0


@dataclass(frozen=True)
class SeedResult:
    outcome: Outcome
    plan: SeedPlan
    error: str | None = None

    @property
    def status_line(self) -> str:
        return " ".join(
            (
                "startup_seed:",
                f"cache={self.plan.state.value}",
                f"present={self.plan.cached_pairs}/{self.plan.total_pairs}",
                f"unreadable={self.plan.unreadable_pairs}",
                f"action={self.plan.action.value}",
                f"outcome={self.outcome.value}",
                f"reason={self.plan.reason}",
                f"scope={self.plan.config.from_year}-{self.plan.config.to_year}",
                f"weeks={len(self.plan.config.weeks)}",
                f"estimated_calls={self.plan.estimated_calls}",
            )
        )


# ---------------------------------------------------------------- configuration


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    """Strict, and only `true`/`false` count.

    A flag whose spelling is wrong must not fall back to the default: `CFB_SEED_
    TEAM_STATS=1` meaning "on" when the operator meant "off" is a silent bill, and
    this project has already been bitten by exactly that shape in the Azure deploy
    gate (`vars.DEPLOY_AZURE != 'false'` -- see tests/test_deploy_workflow.py). The
    message names the variable, because an error that does not is useless to whoever
    is reading a log at 3am.
    """
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false', got {raw!r}")


def _parse_int(name: str, raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_weeks(raw: str | None) -> tuple[int, ...]:
    """`1-16` or `1,2,3` or a mix. An out-of-range week is a billed empty call.

    Duplicates are folded rather than refused. `scripts/backfill_team_stats.py`
    rejects them because it hands the list straight to the fetcher, which
    appends the cached frame once per occurrence and doubles every row. Here the
    week list only ever builds a set of (season, week) pairs, so a repeat is not
    a second bill -- what must not happen is an *inflated estimate*, since that
    is the number an operator uses to decide whether they can afford the run.
    """
    if raw is None or not raw.strip():
        return DEFAULT_WEEKS
    weeks: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"CFB_SEED_WEEKS range must look like '1-16', got {part!r}")
            try:
                low, high = int(bounds[0]), int(bounds[1])
            except ValueError as exc:
                raise ValueError(f"CFB_SEED_WEEKS range must be integers, got {part!r}") from exc
            if low > high:
                raise ValueError(f"CFB_SEED_WEEKS range {part!r} runs backwards")
            weeks.update(range(low, high + 1))
        else:
            try:
                weeks.add(int(part))
            except ValueError as exc:
                raise ValueError(f"CFB_SEED_WEEKS must be integers, got {part!r}") from exc
    if not weeks:
        raise ValueError("CFB_SEED_WEEKS selected no weeks")
    out_of_range = sorted(w for w in weeks if not MIN_WEEK <= w <= MAX_WEEK)
    if out_of_range:
        raise ValueError(
            f"CFB_SEED_WEEKS must be between {MIN_WEEK} and {MAX_WEEK} (CFBD's parameter "
            f"range); got {out_of_range}. Each would be a billed call that returns nothing."
        )
    return tuple(sorted(weeks))


def config_from_env(env: Mapping[str, str] | None = None) -> SeedConfig:
    """Read the scope from the environment, and never raise.

    A malformed variable disables seeding and names itself in `error`, rather than
    raising into a boot path or being silently coerced into a different scope than
    the operator wrote. The container must come up either way.
    """
    env = os.environ if env is None else env
    try:
        enabled = _parse_bool("CFB_SEED_TEAM_STATS", env.get("CFB_SEED_TEAM_STATS"), True)
        from_year = _parse_int("CFB_SEED_FROM_YEAR", env.get("CFB_SEED_FROM_YEAR"), DEFAULT_FROM_YEAR)
        to_year = _parse_int("CFB_SEED_TO_YEAR", env.get("CFB_SEED_TO_YEAR"), DEFAULT_TO_YEAR)
        weeks = _parse_weeks(env.get("CFB_SEED_WEEKS"))
        resume_partial = _parse_bool(
            "CFB_SEED_RESUME_PARTIAL", env.get("CFB_SEED_RESUME_PARTIAL"), False
        )
    except ValueError as exc:
        return SeedConfig(
            enabled=False,
            from_year=DEFAULT_FROM_YEAR,
            to_year=DEFAULT_TO_YEAR,
            weeks=DEFAULT_WEEKS,
            resume_partial=False,
            error=str(exc),
        )
    if from_year > to_year:
        return SeedConfig(
            enabled=False,
            from_year=from_year,
            to_year=to_year,
            weeks=weeks,
            resume_partial=resume_partial,
            error=(
                f"CFB_SEED_FROM_YEAR {from_year} is after CFB_SEED_TO_YEAR {to_year}, so the "
                "scope is empty. Not seeding: guessing a scope here would spend metered quota "
                "on seasons nobody asked for."
            ),
        )
    return SeedConfig(
        enabled=enabled,
        from_year=from_year,
        to_year=to_year,
        weeks=weeks,
        resume_partial=resume_partial,
    )


# ---------------------------------------------------------------- the cache


def cache_path(cache_dir, season: int, week: int):
    """Where the fetcher keeps one (season, week).

    The convention is `data/team_stats.py::_week_cache_path`'s, restated here so
    that the decision can be made without importing the pandas-dependent fetcher
    at module scope. tests/test_startup_seed.py pins the two against each other,
    because a seeder with a second naming scheme would report `cache=empty`
    against a cache the fetcher considers full and re-bill on every single boot.
    """
    return cache_dir / f"{season}_wk{week:02d}.parquet"


def _default_cache_dir():
    from .data import team_stats

    return team_stats.TEAM_STATS_CACHE_DIR


def _is_usable(path) -> bool:
    """Whether the fetcher would accept this file, asked the fetcher's own way.

    `_read_cache` is the fetch path's staleness gate, so reusing it is what makes
    "the seeder is done" and "the fetcher would re-request this" the same
    statement. A file that exists but fails the gate was already billed once, so
    it must not be re-billed by a boot task.
    """
    from .data import team_stats

    if not path.exists():
        return False
    return team_stats._read_cache(path) is not None


def plan_seed(
    config: SeedConfig,
    cache_dir=None,
    env: Mapping[str, str] | None = None,
) -> SeedPlan:
    """Should this boot spend quota? Pure with respect to the network.

    Reads the cache and the environment, touches nothing else, and never raises:
    a boot path that can throw is a boot path that can crashloop.
    """
    env = os.environ if env is None else env
    cache_dir = _default_cache_dir() if cache_dir is None else cache_dir
    total = len(config.pairs)

    if not config.enabled:
        # An incoherent scope is BLOCKED, not SKIPPED: nothing to do is a different
        # statement from "an operator typo is stopping this and needs a look".
        outcome = Outcome.BLOCKED if config.error else Outcome.SKIPPED
        reason = config.error or "seeding is disabled by CFB_SEED_TEAM_STATS"
        return _plan(Action.SKIP, CacheState.EMPTY, reason, config, 0, 0, total, outcome)

    pairs = config.pairs
    cached = sum(1 for season, week in pairs if _is_usable(cache_path(cache_dir, season, week)))
    unreadable = sum(
        1
        for season, week in pairs
        if cache_path(cache_dir, season, week).exists()
        and not _is_usable(cache_path(cache_dir, season, week))
    )

    if cached == 0 and unreadable == 0:
        state = CacheState.EMPTY
    elif cached == total:
        state = CacheState.COMPLETE
    else:
        state = CacheState.PARTIAL

    if state is CacheState.COMPLETE:
        return _plan(
            Action.SKIP,
            state,
            "every requested season-week is already cached",
            config,
            cached,
            unreadable,
            total,
            Outcome.SKIPPED,
        )

    if state is CacheState.PARTIAL and not config.resume_partial:
        # SKIPPED rather than BLOCKED: the cache is *usable*, just incomplete, and
        # finishing it is a human's money decision. The reason says exactly what to do.
        return _plan(
            Action.SKIP,
            state,
            (
                f"{total - cached} of {total} season-weeks are missing and "
                f"{unreadable} cached file(s) are unreadable. Those files were already paid "
                "for once, so a boot task will not re-bill them unattended. Finish the "
                "backfill deliberately (scripts/backfill_team_stats.py --execute), or set "
                "CFB_SEED_RESUME_PARTIAL=true to have the boot task fill the gap."
            ),
            config,
            cached,
            unreadable,
            total,
            Outcome.SKIPPED,
        )

    if not str(env.get("CFBD_API_KEY", "")).strip():
        return _plan(
            Action.SKIP,
            state,
            (
                "the cache is not populated and CFBD_API_KEY is not set, so there is no way "
                "to fetch. Not attempting: CFBD is metered and a keyless attempt can only "
                "fail. Set CFBD_API_KEY on the container to allow the seed."
            ),
            config,
            cached,
            unreadable,
            total,
            Outcome.BLOCKED,
        )

    if state is CacheState.PARTIAL:
        reason = f"CFB_SEED_RESUME_PARTIAL is set; filling {total - cached} missing season-week(s)"
    else:
        reason = "the cache is empty, so this boot would fetch the whole scope"
    return _plan(
        Action.SEED, state, reason, config, cached, unreadable, total, Outcome.SEEDED
    )


def _plan(action, state, reason, config, cached, unreadable, total, outcome) -> SeedPlan:
    return SeedPlan(
        action=action,
        state=state,
        reason=reason,
        config=config,
        cached_pairs=cached,
        unreadable_pairs=unreadable,
        total_pairs=total,
        outcome=outcome,
    )


# ---------------------------------------------------------------- doing it


def execute_backfill(plan: SeedPlan, env: Mapping[str, str] | None = None) -> None:
    """Spend the quota. The only function in this module that can.

    Re-checks the key itself rather than trusting `plan_seed` to have done it, so
    that no future caller -- a route, a cron, a refactor that drops the check --
    can reach the metered API by accident. The cache-or-fetch call is the same
    `fetch_team_stats` that `scripts/backfill_team_stats.py` uses, so there is one
    fetch path, not two that can drift.
    """
    env = os.environ if env is None else env
    if not str(env.get("CFBD_API_KEY", "")).strip():
        raise RuntimeError(
            "CFBD_API_KEY is not set: refusing to call the metered CFBD API. "
            "(A dry run needs no key; the boot task only fetches when a key is present.)"
        )

    from .data import team_stats

    team_stats.fetch_team_stats(plan.config.seasons, weeks=list(plan.config.weeks))


def run_startup_seed(
    config: SeedConfig | None = None,
    cache_dir=None,
    executor: Callable[[SeedPlan], None] | None = None,
    env: Mapping[str, str] | None = None,
) -> SeedResult:
    """Decide, act at most once, and report. Never raises.

    One call, no retry, no `except`-and-try-again. A retry here multiplies the
    bill on precisely the failure most likely to be transient (CFBD unreachable),
    and an escaping exception turns the container into a restart loop that
    re-attempts the bill on every restart. Both are worse than a logged failure
    and a server that comes up.
    """
    env = os.environ if env is None else env
    config = config_from_env(env) if config is None else config
    executor = execute_backfill if executor is None else executor

    plan = plan_seed(config, cache_dir=cache_dir, env=env)

    if not plan.should_seed:
        result = SeedResult(outcome=plan.outcome, plan=plan)
        _log(result)
        return result

    try:
        executor(plan)
    except Exception as exc:  # noqa: BLE001 - a boot task must not kill the process
        result = SeedResult(outcome=Outcome.FAILED, plan=plan, error=f"{type(exc).__name__}: {exc}")
        _log(result)
        return result

    result = SeedResult(outcome=Outcome.SEEDED, plan=plan)
    _log(result)
    return result


def _log(result: SeedResult) -> None:
    logger.info(
        "%s%s", result.status_line, f" error={result.error}" if result.error else ""
    )


# ---------------------------------------------------------------- entrypoint


def main(
    argv: Sequence[str] | None = None,
    config: SeedConfig | None = None,
    cache_dir=None,
    executor: Callable[[SeedPlan], None] | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """The container's boot task. Always returns 0.

    Non-zero here would be chained into the server command and put the container
    into a crashloop, which re-runs this task -- and so re-attempts the bill -- on
    every restart. A seeding failure is a log line, not a boot failure.

    `--check` reports the decision and stops, which is the safe way for an
    operator to ask "is my cache populated?" without any possibility of a bill.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cfb_predictor.startup_seed",
        description=(
            "Seed the CFBD team-stats cache on first boot. Idempotent: a populated cache "
            "is a no-op. Costs one metered CFBD call per (season, week) in scope."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the cache state and the decision, and fetch nothing",
    )
    args = parser.parse_args(argv)

    if args.check:
        env = os.environ if env is None else env
        config = config_from_env(env) if config is None else config
        plan = plan_seed(config, cache_dir=cache_dir, env=env)
        # A check that *would* seed is reported as blocked: nothing was fetched, and
        # the line is read by whoever is deciding whether to go and spend the quota.
        result = SeedResult(
            outcome=Outcome.SKIPPED if not plan.should_seed else Outcome.BLOCKED,
            plan=plan,
        )
        print(f"{result.status_line} (check only, nothing was fetched)", flush=True)
        _log(result)
        return 0

    result = run_startup_seed(config=config, cache_dir=cache_dir, executor=executor, env=env)
    print(result.status_line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
