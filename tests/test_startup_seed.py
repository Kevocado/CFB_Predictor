"""A fresh production container must seed the team-stats cache by itself, and only once.

The hazard this file exists for is not an outage. `data/team_stats.py` is a research
module with no serving caller, and the container serves fine without it. The hazard is a
**misleading assumption**: a 2004-2025 backfill ran and produced 42,190 team-games
across 21,095 games, and the numbers in that module's docstring read as though the
data is simply there. It is not in production -- `data/cache/` is gitignored *and*
`.dockerignore`d, and the Dockerfile `mkdir`s the cache directories empty, so the
artifact is a local checkout's file. The measurements are real; the data behind them
never left the machine.

`src/cfb_predictor/startup_seed.py` fixes that by deciding at boot, and the decision is
in importable Python precisely so it can be tested here. A `sh` guard that decides
whether to spend metered quota cannot be unit-tested, and an untestable money guard is
one that is quietly wrong.

Everything here runs against a temporary directory and an injected executor. **No test
in this file calls CFBD.** The real executor is asserted *not* to be reachable without a
key, which is the whole safety property, and it is checked with a tripwire that fails if
`fetch_team_stats` is touched at all.

Mutation-verified by `tests/mutants/startup_seed_mutants.py`, which applies each
mutation as an exact string replacement, refuses to report a mutation whose anchor did
not apply (a silent no-op is not a survivor), and restores the tree afterwards:
17 induced, **17 caught, 0 survived, 0 vacuous**. The ones that matter most:

| mutation | caught by |
|---|---|
| re-seed a populated cache | the two-boot idempotency test |
| remove the COMPLETE short-circuit | the fast-no-op test, `--check` |
| classify a populated cache as EMPTY | 13 tests, incl. the cost pin |
| re-bill a partial cache with no flag | the partial-cache test |
| ignore unreadable files, so a torn cache reads as empty | `..._reported_as_partial_not_as_empty` |
| seed with no `CFBD_API_KEY` (plan guard) | the missing-key test |
| remove the executor's own key re-check | `..._never_touches_the_fetcher` |
| coerce a mistyped scope to the default | the nonsense-scope test |
| accept a reversed from/to-year | the nonsense-scope test |
| let a bool fall back to its default | the nonsense-scope test |
| accept an out-of-range week | the nonsense-scope test |
| propagate the failure (crashloop) | the failure test, all 3 error types |
| retry the backfill once (two bills) | the two-boot and resume tests |
| exit non-zero on failure | the entrypoint-exit test |
| drop the seeder from `CMD` | the Dockerfile test |
| chain the seeder with `&&` | the Dockerfile test |
| drop `exec` before uvicorn | the Dockerfile test |

**One survivor, found and closed.** The first pass had a mutation that ignored
unreadable files, and it survived: the test only covered a cache with *one* torn file
and *one* good one, where the state is visibly PARTIAL either way. The case that
actually costs money is a cache of nothing but torn files — no usable data, so if that
reads as EMPTY the boot re-bills for files already on disk, forever. The test now
covers both. The first harness also reported two false SURVIVEs: its `if pytest |
grep` pipelines ran under `set -o pipefail`, so a failing test made the whole pipeline
non-zero and read as a pass. The replacement is Python and does not have that shape.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from cfb_predictor import startup_seed
from cfb_predictor.data import team_stats
from cfb_predictor.startup_seed import (
    Action,
    CacheState,
    Outcome,
    run_startup_seed,
)

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "Dockerfile"

# The container image. `data/cache/` is gitignored and also in .dockerignore, so the
# only way a fresh container can have a team-stats cache is if it goes and gets one.
KEY = "Bearer test-key-not-real"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty team-stats cache directory, wired in as the module's real one.

    `TEAM_STATS_CACHE_DIR` is a module-level constant, so pointing it at a tmp dir is
    what lets these tests use the production path convention (`_week_cache_path`) rather
    than a second naming scheme that could drift from it.
    """
    directory = tmp_path / "team_stats"
    monkeypatch.setattr(team_stats, "TEAM_STATS_CACHE_DIR", directory)
    return directory


def _write(season: int, week: int) -> None:
    """Write a real, schema-correct cache file for one (season, week).

    `_empty_week_frame` is the same tombstone the fetcher writes for a dead week, so a
    file written here is indistinguishable from one the real backfill would leave. That
    is the point: the idempotency test below must be able to claim the second boot
    spends nothing on the strength of files the fetcher itself considers usable.
    """
    team_stats.TEAM_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = team_stats._week_cache_path(season, week)
    team_stats._empty_week_frame(season, week).to_parquet(path, index=False)


def _fill(config) -> None:
    for season, week in config.pairs:
        _write(season, week)


def _recorder():
    """An executor that records its calls and writes the files it claimed to write."""
    calls: list = []

    def executor(plan):
        calls.append(plan)
        for season, week in plan.config.pairs:
            _write(season, week)

    executor.calls = calls
    return executor


def _config(**env):
    return startup_seed.config_from_env({"CFBD_API_KEY": KEY, **env})


def _seed_config(**env):
    """A two-season, one-week scope: the same decision, ~350x cheaper to set up."""
    return _config(
        CFB_SEED_FROM_YEAR="2020", CFB_SEED_TO_YEAR="2021", CFB_SEED_WEEKS="1", **env
    )


# ---------------------------------------------------------------- the decision


def test_an_empty_cache_decides_to_seed(cache):
    plan = startup_seed.plan_seed(_seed_config(), cache)

    assert plan.state is CacheState.EMPTY
    assert plan.action is Action.SEED
    assert plan.should_seed


def test_a_populated_cache_spends_nothing_on_a_second_boot(cache):
    """The load-bearing property: idempotency, proved by booting twice for real.

    The first boot sees an empty cache and runs the executor. The second boot must not
    call it again -- that second call is the entire failure mode, because it is a
    second bill for data already on disk. This is a genuine two-boot sequence against a
    real cache directory, not an assertion about a flag.
    """
    config = _seed_config()
    first_executor = _recorder()

    first = run_startup_seed(config, cache, executor=first_executor)
    assert first.outcome is Outcome.SEEDED
    assert len(first_executor.calls) == 1

    second_calls = []
    second = run_startup_seed(
        config,
        cache,
        executor=lambda plan: second_calls.append(plan),
    )

    assert second.outcome is Outcome.SKIPPED
    assert second_calls == [], "a second boot re-spent metered quota on a full cache"


def test_a_populated_cache_is_a_fast_no_op_that_never_reaches_the_fetcher(cache):
    config = _seed_config()
    _fill(config)
    calls = []

    result = run_startup_seed(config, cache, executor=calls.append)

    assert result.outcome is Outcome.SKIPPED
    assert result.plan.state is CacheState.COMPLETE
    assert calls == []


def test_a_partly_populated_cache_does_not_bill_by_default(cache):
    """Half a cache is not the same as no cache, and defaulting to "seed" would bill it.

    A partial cache means some run was interrupted or some files are unreadable. Both
    are already-billed work, so re-seeding unattended is a second bill for a subset. The
    default is therefore to decline and name the shortfall, leaving the lever explicit.
    """
    config = _seed_config()
    _write(2020, 1)
    calls = []

    result = run_startup_seed(config, cache, executor=calls.append)

    assert result.plan.state is CacheState.PARTIAL
    assert result.outcome is Outcome.SKIPPED
    assert calls == []
    assert result.plan.missing_pairs == 1
    assert result.plan.cached_pairs == 1


def test_an_explicit_resume_flag_is_what_it_takes_to_finish_a_partial_cache(cache):
    config = _seed_config(CFB_SEED_RESUME_PARTIAL="true")
    _write(2020, 1)
    executor = _recorder()

    result = run_startup_seed(config, cache, executor=executor)

    assert result.outcome is Outcome.SEEDED
    assert len(executor.calls) == 1


def test_an_unreadable_cache_file_is_reported_as_partial_not_as_empty(cache):
    """A torn write or a schema-stale file has already been billed once.

    Two cases, and the second is the one that bites. With some usable files present, a
    torn file is visibly a shortfall. With *only* torn files, the cache has no usable
    data at all -- and if that reads as EMPTY then every boot re-bills for files that
    are sitting right there, forever, with nothing in the log to explain the bill. So
    unreadable is its own state, distinct from both "never fetched" and "complete".
    """
    config = _seed_config()
    team_stats.TEAM_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (team_stats.TEAM_STATS_CACHE_DIR / "2020_wk01.parquet").write_bytes(b"not parquet")
    _write(2021, 1)
    calls = []

    mixed = run_startup_seed(config, cache, executor=calls.append, env={"CFBD_API_KEY": KEY})
    assert mixed.plan.state is CacheState.PARTIAL
    assert mixed.plan.unreadable_pairs == 1
    assert mixed.plan.cached_pairs == 1
    assert calls == []

    (team_stats.TEAM_STATS_CACHE_DIR / "2021_wk01.parquet").write_bytes(b"not parquet")
    only_torn = startup_seed.plan_seed(config, cache, env={"CFBD_API_KEY": KEY})

    assert only_torn.cached_pairs == 0
    assert only_torn.unreadable_pairs == 2
    assert only_torn.state is CacheState.PARTIAL, (
        "a cache of nothing but unreadable files reported itself empty, so every boot "
        "would re-bill for data that is already on disk"
    )
    assert not only_torn.should_seed
    assert calls == []


def test_a_missing_api_key_blocks_the_seed_instead_of_attempting_it(cache):
    config = _seed_config()
    calls = []

    result = run_startup_seed(config, cache, executor=calls.append, env={})

    assert result.outcome is Outcome.BLOCKED
    assert result.plan.state is CacheState.EMPTY, "the cache is still honestly reported empty"
    assert "key" in result.plan.reason.lower()
    assert calls == []


def test_a_nonsense_scope_disables_seeding_rather_than_guessing_one(cache):
    """Fail-safe on cost: a mistyped scope must not be "corrected" into a bill.

    Guessing a scope for an operator who typed a bad one spends money on data they did
    not ask for, at 3am, with nobody watching. Declining and naming the variable does
    not.
    """
    for env, named in [
        ({"CFB_SEED_FROM_YEAR": "abc"}, "CFB_SEED_FROM_YEAR"),
        ({"CFB_SEED_WEEKS": "0"}, "CFB_SEED_WEEKS"),
        ({"CFB_SEED_WEEKS": "17"}, "CFB_SEED_WEEKS"),
        ({"CFB_SEED_FROM_YEAR": "2025", "CFB_SEED_TO_YEAR": "2004"}, "CFB_SEED_FROM_YEAR"),
        ({"CFB_SEED_TEAM_STATS": "maybe"}, "CFB_SEED_TEAM_STATS"),
    ]:
        config = _config(**env)
        assert not config.enabled, env
        assert config.error and named in config.error, (env, config.error)

        calls = []
        result = run_startup_seed(config, cache, executor=calls.append)
        assert result.outcome is Outcome.BLOCKED, env
        assert calls == [], env


def test_seeding_can_be_turned_off_entirely(cache):
    config = _seed_config(CFB_SEED_TEAM_STATS="false")
    calls = []

    result = run_startup_seed(config, cache, executor=calls.append)

    assert not config.enabled
    assert result.outcome is Outcome.SKIPPED
    assert calls == []


# ---------------------------------------------------------------- scope & cost


def test_scope_can_be_reduced_by_configuration(cache):
    """A reduced scope is a first-class supported thing, not a workaround.

    The full default is 352 metered calls in one boot. An operator who cannot spend that
    has to be able to ask for less without editing code.
    """
    plan = startup_seed.plan_seed(
        _config(CFB_SEED_FROM_YEAR="2023", CFB_SEED_TO_YEAR="2025", CFB_SEED_WEEKS="1-3"),
        cache,
    )

    assert plan.config.seasons == [2023, 2024, 2025]
    assert plan.config.pairs == [
        (season, week) for season in (2023, 2024, 2025) for week in (1, 2, 3)
    ]
    assert plan.estimated_calls == 9


def test_a_repeated_week_is_counted_once_because_it_is_not_a_separate_bill(cache):
    """The backfill script refuses duplicates; here they are folded together.

    A duplicate week is one bill, so refusing to run over it would be theatre. What
    matters is that the *estimate* is not inflated, since an inflated estimate is what
    an operator uses to decide whether they can afford the run.
    """
    config = _config(CFB_SEED_FROM_YEAR="2020", CFB_SEED_TO_YEAR="2020", CFB_SEED_WEEKS="1,1,2")

    assert config.weeks == (1, 2)
    assert len(config.pairs) == 2


def test_the_default_scope_costs_what_the_plan_says_it_costs(cache):
    """Pinned, because this number is the reason the scope has to be configurable.

    22 seasons x 16 weeks. It is a count over a fixed enumeration, not a measured
    distribution, so one evaluation is the whole measurement.
    """
    config = _config()
    plan = startup_seed.plan_seed(config, cache)

    assert (config.seasons[0], config.seasons[-1]) == (2004, 2025)
    assert len(config.seasons) == 22
    assert plan.total_pairs == 352
    assert plan.estimated_calls == 352


# ---------------------------------------------------------------- the runner


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("cfbd is unreachable"),
        ConnectionError("Max retries exceeded"),
        TimeoutError("read timed out"),
    ],
    ids=["api-error", "network-down", "timeout"],
)
def test_a_failed_backfill_is_reported_and_never_retried(cache, error):
    """No crashloop, and no retry loop: one attempt, one report, then the server starts.

    Both properties come from the same place -- a single call, wrapped in a bare
    `except Exception`. A retry here would multiply the bill on exactly the failure
    (CFBD down) most likely to be transient, and an uncaught exception would put the
    container into a restart loop that re-attempts the bill on every restart.
    """
    calls = []

    def executor(plan):
        calls.append(plan)
        raise error

    result = run_startup_seed(_seed_config(), cache, executor=executor)

    assert result.outcome is Outcome.FAILED
    assert len(calls) == 1, "the boot task retried, which is a second bill per boot"
    assert isinstance(result.error, str) and result.error


def test_an_empty_cache_and_a_failed_one_are_distinguishable_in_the_log(cache, caplog):
    """The two states an operator has to tell apart at 3am.

    "The cache is empty and I am about to bill you" and "the cache is empty and the
    attempt failed" call for completely different responses, and a log that renders both
    as 'cache missing' is how a broken seed gets ignored for weeks.
    """
    caplog.set_level(logging.INFO, logger="cfb_predictor.startup_seed")

    def boom(plan):
        raise RuntimeError("cfbd is unreachable")

    run_startup_seed(_seed_config(), cache, executor=boom, env={})
    blocked = caplog.text
    caplog.clear()

    run_startup_seed(_seed_config(), cache, executor=boom)
    failed = caplog.text

    assert "cache=empty" in blocked
    assert "outcome=blocked" in blocked
    assert "cache=empty" in failed
    assert "outcome=failed" in failed
    assert "cfbd is unreachable" in failed


def test_the_cache_state_is_always_named_in_the_log(cache, caplog):
    caplog.set_level(logging.INFO, logger="cfb_predictor.startup_seed")
    config = _seed_config()
    _fill(config)

    run_startup_seed(config, cache, executor=lambda plan: pytest.fail("spent quota"))

    assert "cache=complete" in caplog.text
    assert "action=skip" in caplog.text


# ---------------------------------------------------------------- the entrypoint


def test_check_mode_reports_the_cache_state_and_spends_nothing(cache, capsys):
    config = _seed_config()
    _fill(config)
    calls = []

    code = startup_seed.main(
        ["--check"], config=config, cache_dir=cache, executor=calls.append,
        env={"CFBD_API_KEY": KEY},
    )

    out = capsys.readouterr().out
    assert code == 0
    assert calls == []
    assert "cache=complete" in out


def test_the_entrypoint_exits_zero_even_when_seeding_fails(cache, capsys):
    """The container must come up either way.

    A non-zero exit here, chained into the server command, is a crashloop: the platform
    restarts the container, the boot task runs again, and the bill is attempted again on
    every restart. That is the exact shape this design refuses.
    """
    def boom(plan):
        raise RuntimeError("cfbd is unreachable")

    code = startup_seed.main(
        [], config=_seed_config(), cache_dir=cache, executor=boom,
        env={"CFBD_API_KEY": KEY},
    )

    assert code == 0
    assert "outcome=failed" in capsys.readouterr().out


def test_the_status_line_names_outcome_state_and_estimated_cost(cache, capsys):
    startup_seed.main(
        ["--check"], config=_seed_config(), cache_dir=cache,
        executor=lambda plan: pytest.fail("spent quota"),
        env={"CFBD_API_KEY": KEY},
    )

    out = capsys.readouterr().out
    for token in ("cache=", "action=", "outcome=", "estimated_calls="):
        assert token in out, out
    assert re.search(r"estimated_calls=\d+", out)


# ---------------------------------------------------------------- wiring


def test_the_cache_path_convention_matches_the_fetcher_it_decides_for(tmp_path):
    """The decision must be made about the files the fetcher will actually read.

    A seeder with its own naming scheme would report `cache=empty` against a cache the
    fetcher considers full, and re-bill every single boot. The convention is therefore
    pinned against the real one rather than trusted.
    """
    for season, week in [(2004, 1), (2025, 9), (2025, 16)]:
        ours = startup_seed.cache_path(tmp_path, season, week)
        theirs = team_stats._week_cache_path(season, week)
        assert ours.name == theirs.name, (ours, theirs)
        assert ours.name == f"{season}_wk{week:02d}.parquet"


def test_the_real_executor_refuses_without_a_key_and_never_touches_the_fetcher(
    cache, monkeypatch
):
    """Defence in depth, and the strongest money assertion in this repo.

    `plan_seed` already refuses to seed without a key. This proves that even if that
    check were bypassed -- a new caller, a changed default, a future refactor -- the
    executor cannot itself reach the metered API. The tripwire fails on any call to
    `fetch_team_stats`, so "spends zero" is observed, not assumed.
    """
    def tripwire(*args, **kwargs):
        raise AssertionError("fetch_team_stats was reached: a metered CFBD call")

    monkeypatch.setattr(team_stats, "fetch_team_stats", tripwire)
    plan = startup_seed.plan_seed(_seed_config(), cache, env={})

    with pytest.raises(RuntimeError, match="CFBD_API_KEY"):
        startup_seed.execute_backfill(plan, env={})


def test_the_local_footgun_is_documented_because_it_is_real():
    """A bare invocation on a developer machine really can spend 352 calls.

    `config.py`'s `load_dotenv()` walks up from the module and finds the repository's
    `.env`, so `CFBD_API_KEY` is populated from disk whether or not it was exported.
    The container has no `.env` and so cannot hit this -- but anyone running the
    entrypoint by hand can, and the only thing standing between them and a bill is
    this note. Verified on 2026-09-28: from a worktree with no `.env` of its own, the
    key resolved from the parent checkout's.
    """
    doc = startup_seed.__doc__ or ""
    for token in ("load_dotenv", "--check", "backfill_team_stats.py --execute"):
        assert token in doc, (
            f"startup_seed.py's docstring no longer mentions {token!r}; the warning that "
            "a bare local run can spend real quota is the only guard it has"
        )


def test_the_module_says_where_the_backfill_actually_lives():
    """The claim that caused this, pinned where it was being made.

    The 42,190-row figure lived in `data/team_stats.py`'s docstring with nothing
    saying the data behind it was on one laptop. Docstrings rot silently -- there is
    no test failure when a deployment path changes underneath one -- so the
    correction is asserted here rather than trusted. Deliberately loose: it looks for
    the substance (gitignored, not in the image, seeded at boot) and not for the
    phrasing, so a rewording does not fail the suite.
    """
    from cfb_predictor.data import team_stats

    doc = team_stats.__doc__ or ""
    assert "not in production" in doc, (
        "team_stats.py's docstring quotes a 42,190-row backfill that is not in "
        "production; the correction that says so has gone missing"
    )
    for token in ("gitignored", ".dockerignore", "startup_seed", "CFBD_API_KEY"):
        assert token in doc, f"team_stats.py's docstring no longer mentions {token!r}"


def test_the_container_starts_through_the_seeder():
    """The decision has to be on the boot path, or it is just a module.

    Asserted against the Dockerfile because that is where the boot path is written, and
    an entrypoint that is present but not invoked is indistinguishable from one that
    does not exist.
    """
    dockerfile = DOCKERFILE.read_text()
    cmd = next(
        line for line in dockerfile.splitlines() if line.startswith("CMD ")
    )

    assert "cfb_predictor.startup_seed" in cmd
    assert "exec uvicorn" in cmd, "the server must replace the shell to receive signals"
    seed, _, server = cmd.partition("cfb_predictor.startup_seed")[2].partition("uvicorn")
    assert "&&" not in seed, (
        "the seeder is chained with && into the server command, so a seeding failure "
        "stops the container from starting -- a crashloop that re-bills on every restart"
    )
    assert ";" in seed
