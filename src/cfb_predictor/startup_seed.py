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

The status line
---------------
One line, in a fixed order, and **every field on it is about the run, not about
the run's intentions.** That distinction is not cosmetic; it was the defect.

    startup_seed: cache=... present=.../... unreadable=... action=... \
[plan_reason=... ]outcome=... reason=... [error=... ]scope=... weeks=... \
estimated_calls=... [cached_now=.../... unreadable_now=... ]

- `present`/`unreadable` count the cache **as the decision saw it**, before
  anything was fetched. `cached_now`/`unreadable_now` count it **after**, and
  appear only if a fetch was actually attempted. A run that wrote 91 of 299
  files and then raised has to be able to say so, and the only way to say it is
  to have measured both ends.
- `plan_reason` is the *decision's* rationale -- why the seeder chose the action
  it chose. On a resume that is "CFB_SEED_RESUME_PARTIAL is set; filling 299
  missing season-week(s)". It is printed only when a fetch was attempted, i.e.
  only when the run could diverge from the decision; otherwise `reason` already
  says it and the line would say everything twice.
- `reason` is the *outcome's* rationale -- why the run ended the way it did --
  and it is the field an operator reads first. It used to be the plan's rationale
  wearing the wrong label, which produced the contradiction seen in production on
  2026-09-28: a line reading `action=seed outcome=failed` whose `reason` described
  seeding as though it were still under way, and which carried no error at all.
- `error` is on the **line itself**, not only in the log record. It used to be
  appended by `_log` alone, and `_log` emits through `logger.info` -- and nothing
  in this package configures logging, so in the boot process (a bare
  `python -m cfb_predictor.startup_seed`, before uvicorn exists) `root` has no
  handlers, the effective level is WARNING, and `logging.lastResort` is
  WARNING-level. `_log` therefore printed **nothing**. The exception text was
  computed, stored on the result, and dropped on the floor, while the one channel
  that does reach the operator -- `main`'s `print` -- had no field for it.

A "failed" line here is *true*, and must not be softened: `fetch_team_stats`
writes each (season, week) as it goes, so a run that raises partway leaves
already-paid-for files behind and does not finish its scope. What was untrue was
the silence around it -- no error, and a `reason` that pointed at the plan rather
than the failure.

Configuration (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `CFB_SEED_TEAM_STATS` | `true` | master switch |
| `CFB_SEED_FROM_YEAR` | `2004` | first season |
| `CFB_SEED_TO_YEAR` | `2025` | last season |
| `CFB_SEED_WEEKS` | `1-16` | `1-16`, or a list like `1,2,3` |
| `CFB_SEED_RESUME_PARTIAL` | `false` | seed a partly-populated cache |
| `CFBD_API_KEY` | -- | required before any call is attempted |

**Set-but-blank is an error, not a default.** `CFB_SEED_WEEKS=""` used to read as
unset and fall back to `1-16` -- 352 metered calls from a variable an operator
believed they had narrowed to nothing. `CFB_SEED_TEAM_STATS=""` read as *on*.
Every parser here now refuses a present-but-whitespace value and names itself, so
the failure mode is `outcome=blocked` with the variable in `reason`, not a bill.
It matters in compose, where the common `FOO: ${FOO:-}` idiom turns "unset" into
"empty string" rather than leaving it unset.

A variable set while seeding is *off* is reported as ignored, by name, in the
same line.

The default is the full 2004-2025 sweep because that is the population
`data/team_stats.py`'s measurements are stated over; narrowing it silently
would leave the docstring and the cache describing different things, which is
the exact confusion this module is here to remove. If 352 calls is not
affordable, set the scope rather than accepting a partial cache silently.

Running it by hand
------------------
`--check` reports the decision and fetches nothing. Use it first.

**On a developer machine, a bare `python -m cfb_predictor.startup_seed` can
really spend 352 calls**, because `config.py`'s `load_dotenv()` walks up from
this file and finds the repository's `.env`, so `CFBD_API_KEY` is populated from
disk even when you never exported it. (In the container there is no `.env` --
the Dockerfile copies `src/`, `models/` and one data file and nothing else -- so
the key can only arrive by being injected, which is the intended deployment
act.) To seed a checkout deliberately, use the script that requires a
deliberate flag and prints its plan first:
`scripts/backfill_team_stats.py --execute`.

**That script does not exist inside the container**, and advice that names a
path the reader cannot reach is worse than no advice: the Dockerfile copies
`pyproject.toml`, `src/`, `models/` and `data/public_snapshot.json` and nothing
else, so `/app` holds exactly `data models pyproject.toml src` and
`scripts/backfill_team_stats.py` fails with `can't open file`. The lever inside
the container is this module, run deliberately:

    docker compose exec -e CFB_SEED_RESUME_PARTIAL=true cfb python -m cfb_predictor.startup_seed

**`CFB_SEED_RESUME_PARTIAL` must be passed with `-e`, and putting it in
`/opt/stack/.env` does nothing.** Compose forwards a variable into a container
only if the service's `environment:` block names it, and the `cfb` service's
block does not list `CFB_SEED_RESUME_PARTIAL` (nor any other `CFB_SEED_*`), so
`.env` is used for compose's own interpolation and never reaches the process.
An operator who sets it there, restarts, and sees `action=skip` will conclude
the lever is broken; it is not, it was never delivered. `docker compose exec -e`
injects it for that one process, which is what the command above does.

**This module cannot warn about that particular mistake, and saying so is more
useful than pretending otherwise.** It reads `os.environ`; a variable compose
never forwarded is, by definition, invisible to it, and there is no configuration
layer that can report on a value it was never given. The `.env` case is one of
only two ways a set variable can be ignored from the inside -- the other, which
*is* detectable and is handled, is a variable set while seeding is off (named in
the reason) or set blank (refused outright). So the guidance above lives in the
*message* the operator sees, where it can be acted on, rather than in a check
that would have to be a guess.

Fixing the forwarding itself means adding `CFB_SEED_*` to the `cfb` service's
`environment:` block in the VPS stack's `compose.yml`. That is a different
repository, and it is deliberately not done here.
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

# Everything this module reads except the key. If the master switch is off, every
# one of these is set-but-ignored, and the log says so by name rather than
# leaving the operator to wonder why their carefully-chosen scope did nothing.
SCOPE_VARS = (
    "CFB_SEED_FROM_YEAR",
    "CFB_SEED_TO_YEAR",
    "CFB_SEED_WEEKS",
    "CFB_SEED_RESUME_PARTIAL",
)


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
    """What happened, as distinct from what was decided.

    `plan.reason` explains the *decision* and is not a report of the run. So the
    status line carries both, under the labels `plan_reason=` and `reason=`, and
    only the second is allowed to explain `outcome=`. Folding them together is
    what let a real failure render as a contradiction -- `action=seed
    outcome=failed` with a `reason` describing a seed that was still to come --
    and it is the reason they are separate fields now.

    `reason` is an override, not a computation, so that the one caller which did
    not act on the plan (`--check`) can say *that* instead of inheriting a
    rationale for something it never did. Left `None`, the outcome's own
    rationale is derived.

    `error` is rendered onto the line rather than only logged. `_log` emits
    through `logger.info`, and nothing in this package configures logging, so in
    the boot process -- a bare `python -m cfb_predictor.startup_seed`, before
    uvicorn exists -- the log record goes nowhere at all: the root logger has no
    handlers, the effective level is WARNING, and `logging.lastResort` is
    WARNING. The error was computed and then discarded on the floor, while the
    only channel that does reach the operator had no field to carry it.

    `cached_after`/`unreadable_after` are the cache re-counted, with the
    fetcher's own gate, once the attempt is over. They are `None` when nothing
    was attempted, and they are what lets an interrupted run report that it
    wrote 91 of the 299 files it planned, rather than only that it stopped.
    """

    outcome: Outcome
    plan: SeedPlan
    error: str | None = None
    reason: str | None = None
    cached_after: int | None = None
    unreadable_after: int | None = None

    @property
    def outcome_reason(self) -> str:
        """Why the run ended the way it did, or `plan.reason` when it never ran.

        For `SKIPPED` and `BLOCKED` nothing was attempted, so the decision's
        rationale *is* the outcome's rationale and saying anything else would
        pad the line. For `SEEDED` and `FAILED` a fetch was attempted, and the
        plan's rationale is silent about how that went -- which is the whole
        defect.
        """
        if self.reason is not None:
            return self.reason
        if self.outcome is Outcome.SEEDED:
            return (
                "the backfill returned without raising, so the scope was fetched as planned"
            )
        if self.outcome is Outcome.FAILED:
            return (
                f"the backfill raised, so the {self.plan.estimated_calls} season-week(s) in "
                f"scope were NOT all written; the "
                f"{self._written} file(s) it did write are kept and the next boot will not "
                f"re-bill them, so this is resumable rather than lost"
            )
        return self.plan.reason

    @property
    def _written(self) -> int:
        if self.cached_after is None:
            return 0
        return max(0, self.cached_after - self.plan.cached_pairs)

    @property
    def status_line(self) -> str:
        attempted = self.cached_after is not None
        fields = [
            "startup_seed:",
            f"cache={self.plan.state.value}",
            f"present={self.plan.cached_pairs}/{self.plan.total_pairs}",
            f"unreadable={self.plan.unreadable_pairs}",
            f"action={self.plan.action.value}",
        ]
        # `plan_reason` earns its place only once a fetch was actually attempted,
        # which is exactly when the run can diverge from the decision. When
        # nothing was attempted -- skipped, blocked, or `--check` -- `reason` *is*
        # the plan's rationale, and printing it twice would double the longest
        # field in the line for no information at all. The advice to finish a
        # partial cache is already long; it is read once.
        if attempted:
            fields.append(f"plan_reason={self.plan.reason}")
        fields += [f"outcome={self.outcome.value}", f"reason={self.outcome_reason}"]
        if self.error:
            fields.append(f"error={self.error}")
        fields += [
            f"scope={self.plan.config.from_year}-{self.plan.config.to_year}",
            f"weeks={len(self.plan.config.weeks)}",
            f"estimated_calls={self.plan.estimated_calls}",
        ]
        if attempted:
            fields += [
                f"cached_now={self.cached_after}/{self.plan.total_pairs}",
                f"unreadable_now={self.unreadable_after}",
            ]
        return " ".join(fields)


# ---------------------------------------------------------------- configuration


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    """Strict, and only `true`/`false` count.

    A flag whose spelling is wrong must not fall back to the default: `CFB_SEED_
    TEAM_STATS=1` meaning "on" when the operator meant "off" is a silent bill, and
    this project has already been bitten by exactly that shape in the Azure deploy
    gate (`vars.DEPLOY_AZURE != 'false'` -- see tests/test_deploy_workflow.py). The
    message names the variable, because an error that does not is useless to whoever
    is reading a log at 3am.

    Blank is refused rather than defaulted, and that is not pedantry:
    `CFB_SEED_WEEKS=""` used to be indistinguishable from `CFB_SEED_WEEKS` being
    absent, so a variable an operator had set to what they plainly meant as
    "nothing" silently became the full `1-16` -- 352 metered calls. Compose makes
    this the common case rather than the exotic one: `CFOO: ${CFOO:-}` resolves
    to an empty string, not to "unset", so every such variable arrives blank.
    A blanked-out `CFB_SEED_TEAM_STATS` reading as *on* is the same silent bill
    in the other direction.
    """
    if raw is None:
        return default
    if not raw.strip():
        raise ValueError(
            f"{name} is set but empty. Refusing to guess: an empty value here would read "
            f"as '{str(default).lower()}'. Unset the variable to take the default, or set "
            "it to 'true' or 'false'."
        )
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be 'true' or 'false', got {raw!r}")


def _parse_int(name: str, raw: str | None, default: int) -> int:
    if raw is None:
        return default
    if not raw.strip():
        raise ValueError(
            f"{name} is set but empty. Refusing to guess: an empty value here would read as "
            f"{default}. Unset the variable to take the default."
        )
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

    Blank is refused for the same reason as in `_parse_bool`, and this is the one
    where it costs the most: an empty `CFB_SEED_WEEKS` defaulted to all sixteen
    weeks, which for the default 22-season scope is the entire 352-call bill for
    a variable that had been set.
    """
    if raw is None:
        return DEFAULT_WEEKS
    if not raw.strip():
        raise ValueError(
            f"CFB_SEED_WEEKS is set but empty. Refusing to guess: an empty value here would "
            f"mean all {len(DEFAULT_WEEKS)} weeks, which is the most expensive reading "
            "available. Unset the variable to take the default, or say what you want "
            "(e.g. '1,2,3')."
        )
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


def _ignored_by_disabled_seeding(env: Mapping[str, str]) -> list[str]:
    """Scope variables that are set right now and are having no effect.

    The other way -- and the only way this process can see -- for an operator to
    set a variable and have it silently do nothing. Compose never forwarding one
    is not in this list, and cannot be: it is not in `os.environ`, so there is
    nothing here to report on. The env-file case is invisible from in here by
    construction, which is why `plan_seed`'s partial-cache reason spells the
    `docker compose exec -e` form out rather than trusting the operator to know.
    """
    return [name for name in SCOPE_VARS if str(env.get(name, "")).strip()]


def _recount(config: SeedConfig, cache_dir) -> tuple[int, int]:
    """`(usable, unreadable)` over the scope, asked the fetcher's own way.

    One place, deliberately. The decision needs this count *before* it acts and
    the status line needs it *after*, and two passes over the same files by two
    different code paths is how the two come to disagree -- a "how much did that
    run actually write" figure computed differently from the "is it done yet"
    figure is worse than either being absent. It reads files and never the
    network, so it is safe to call twice.
    """
    cached = 0
    unreadable = 0
    for season, week in config.pairs:
        path = cache_path(cache_dir, season, week)
        if _is_usable(path):
            cached += 1
        elif path.exists():
            unreadable += 1
    return cached, unreadable


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
        ignored = _ignored_by_disabled_seeding(env)
        if ignored:
            reason += (
                f" (ignored while seeding is off, so these have no effect: "
                f"{', '.join(ignored)})"
            )
        return _plan(Action.SKIP, CacheState.EMPTY, reason, config, 0, 0, total, outcome)

    cached, unreadable = _recount(config, cache_dir)

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
        #
        # **The command named here has to be one that works where this line is
        # read.** This line is read on a container, and `scripts/` is not in the
        # image -- the Dockerfile copies `pyproject.toml`, `src/`, `models/` and
        # `data/public_snapshot.json`, so `/app` holds `data models pyproject.toml
        # src` and nothing else. Telling an operator at 3am to run
        # `scripts/backfill_team_stats.py --execute` inside that container gets
        # them `can't open file`, which reads as "this is broken" rather than as
        # "here is the right way". So the advice names the seeder itself, run
        # through `docker compose exec -e`.
        #
        # **`-e` is load-bearing, and `.env` is not a substitute.** Compose hands a
        # variable to a container only when the service's `environment:` block
        # names it; the `cfb` service's block lists no `CFB_SEED_*` at all. So
        # `CFB_SEED_RESUME_PARTIAL` in `/opt/stack/.env` is consumed by compose's
        # own interpolation, never reaches the process, and a restart still
        # reports `action=skip` -- which looks exactly like a broken lever. This
        # seeder cannot detect that, because it reads `os.environ` and the
        # variable is not in it; the warning has to live in the message.
        return _plan(
            Action.SKIP,
            state,
            (
                f"{total - cached} of {total} season-weeks are missing and "
                f"{unreadable} cached file(s) are unreadable. Those files were already paid "
                "for once, so a boot task will not re-bill them unattended. To finish the "
                "backfill deliberately, from the stack directory on the host, run: "
                "docker compose exec -e CFB_SEED_RESUME_PARTIAL=true cfb "
                "python -m cfb_predictor.startup_seed "
                "(-e injects the variable into that one process; putting "
                "CFB_SEED_RESUME_PARTIAL in the stack's .env does nothing, because the cfb "
                "service does not forward it, so a restart would still report action=skip. "
                "scripts/ is not in the image, so a checkout's backfill script cannot be run "
                "from inside the container.)"
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
    cache_dir = _default_cache_dir() if cache_dir is None else cache_dir

    plan = plan_seed(config, cache_dir=cache_dir, env=env)

    if not plan.should_seed:
        result = SeedResult(outcome=plan.outcome, plan=plan)
        _log(result)
        return result

    try:
        executor(plan)
    except Exception as exc:  # noqa: BLE001 - a boot task must not kill the process
        outcome, error = Outcome.FAILED, f"{type(exc).__name__}: {exc}"
    else:
        outcome, error = Outcome.SEEDED, None

    # Re-read the cache either way. The line has to be able to say how much of
    # the scope is now on disk, because "failed" and "succeeded" are both
    # statements about files, and after a partial write the two are not the same
    # question: a run that stopped at 91 of 299 is resumable, and an operator can
    # only know that from a count taken after the attempt. It reads local files
    # and never the network, so it cannot cost anything.
    cached_after, unreadable_after = _recount(config, cache_dir)
    result = SeedResult(
        outcome=outcome,
        plan=plan,
        error=error,
        cached_after=cached_after,
        unreadable_after=unreadable_after,
    )
    _log(result)
    return result


def _log(result: SeedResult) -> None:
    """Hand the line to `logging`, which is *not* the same as printing it.

    Nothing in this package configures logging, so in the boot process this emits
    nothing: the root logger has no handlers, the effective level is WARNING, and
    `logging.lastResort` is WARNING-level, so an INFO record is dropped. That is
    not a reason to stop calling `_log` -- under uvicorn, or under a host runner
    that configures logging, it is the line an operator gets -- but it *is* why
    the error lives on the status line itself rather than being appended here.
    Whatever appends the diagnosis has to be the same thing that prints the line,
    or a channel that prints is a channel that is missing the one field that
    matters.
    """
    logger.info("%s", result.status_line)


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
        #
        # Every other outcome is the plan's own. The first version of this forced
        # anything that was not going to seed to SKIPPED, which quietly flattened
        # `outcome=blocked` -- a scope nobody can act on, because the variable that
        # says so is misspelled or blank -- into "nothing to do here". The reason
        # still named the variable, so no information was lost, but the one word an
        # operator scans for said the opposite of what it meant.
        #
        # The reason is set explicitly because neither derived one fits. A check
        # has no run to report on, so `plan.reason` -- which describes a seed that
        # did not happen -- would be the plan's rationale masquerading as an
        # outcome's, which is the exact confusion this field split exists to stop.
        if not plan.should_seed:
            reason = f"{plan.reason} (--check: nothing was fetched)"
        else:
            reason = (
                "--check: nothing was fetched; without it this boot would seed "
                f"{plan.estimated_calls} season-week(s) of the scope below"
            )
        result = SeedResult(
            outcome=Outcome.BLOCKED if plan.should_seed else plan.outcome,
            plan=plan,
            reason=reason,
        )
        print(f"{result.status_line} (check only)", flush=True)
        _log(result)
        return 0

    result = run_startup_seed(config=config, cache_dir=cache_dir, executor=executor, env=env)
    print(result.status_line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
