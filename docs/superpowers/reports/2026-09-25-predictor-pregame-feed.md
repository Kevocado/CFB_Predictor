# Step 7a (CFB half): Kalshi pre-game feed — evidence report

- Plan: `docs/superpowers/plans/2026-09-25-predictor-pregame-feed.md` (step 7a, Tasks 5–7)
- Branch: `plan/2026-09-25-kalshi-feed`
- Base: `origin/main` at `4b37b58` (post-merge of PRs #1 and #2)
- NFL half of the same step lives in the NFL repo on a branch of the same name; see its report for
  Tasks 1–4.

## Baseline

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src \
  /Users/sigey/Documents/Projects.nosync/CFB_Predictor/.venv/bin/python -m pytest -q
170 passed
```

The plan's stated baseline (128 tests at `f88fc05`) predates the merges of PRs #1 and #2. The code
anchors the plan's replace-blocks against still matched, so only the counts moved.

## Task 5 — `77d6ce6`: store the snapshot distribution, add the feed + calibration readers

Same change as the NFL half's Task 1, ported: `_connect()` `ALTER TABLE`s in `predicted_margin`,
`sigma`, `predicted_total`, `total_sigma`, `model_version`; `get_feed_predictions()` serves only rows
still ahead of kickoff; `get_calibration()` returns 10-bin `winner` / `spread` / `total` reliability
buckets.

### RED

```text
# re-captured against the parent commit's src/ (see "How the RED evidence was captured")
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_store.py
9 failed, 1 passed in 0.93s
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_store.py
10 passed in 0.44s
```

- **Deviation — no `backfilled` column, no legacy classification pass, no `n_backfilled`.** The plan
  added all three; PR #1 (merged) already shipped pre-game-ness as
  `_snapshotted_after_kickoff(snapshotted_at, commence_time)`, derived live from two columns that
  always existed and failing closed on an unparseable row, and `/api/track-record` already reports
  `n_rebuilt` from that same predicate. Reusing them means the flag cannot drift out of step with the
  timestamps, and a database written before this change needs no migration pass.
- The feed still emits `"backfilled": false`: for a row that passed the filter that is the honest
  statement, and `tradehub/sports/feed.py::parse_feed` rejects a truthy value.

## Task 6 — `b4bcc11`: freeze each snapshot inside a 48 h lead window, record `model_version`

### RED

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_snapshot_window.py
9 failed in 2.38s
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_snapshot_window.py
9 passed in 2.53s
```

- Port of the NFL half's Task 2 with the two CFB-specific interfaces: `games_data.fetch_upcoming_games`
  in place of `schedules.fetch_upcoming_games`, and CFBD's tz-aware UTC `gameday` (so no
  `.replace(tzinfo=None)` step, unlike nflverse's naive dates).
- `SNAPSHOT_LEAD_HOURS = float(os.getenv("SNAPSHOT_LEAD_HOURS", "48"))`;
  `_games_to_snapshot()` concatenates this week's and next week's games and keeps those inside the
  window. `current_season_and_week()` rolls over on the date of week 1's first kickoff, so "this week"
  alone left Friday-night games unsnapshotted until after kickoff (where `record_game_predictions`
  rejects them, so they were never tracked at all) and froze next week's games on the previous
  Saturday, before that day's results.
- `manifest.model_version()` is `f"{manifest['chosen_candidate']}@{manifest['trained_at']}"`, surfaced
  through `load_models()` and copied into every snapshot by `_predict_game_from_models`.
- **Deviation — each snapshot row carries its own `week`.** The tick now snapshots two weeks, so
  labelling every row with the tick's `week` would put next week's games in the wrong week
  everywhere downstream. `int(game["week"]) if pd.notna(game.get("week")) else week`.
- **Kept from the plan, unlike the NFL half — the model load stays at the top of the tick.** The CFB
  tick loads models inside a `try/except` and returns early on failure (pre-existing behaviour the
  plan preserved), so on CFB a model-load failure also skips reconcile and backfill for that tick,
  whereas the NFL half loads models only when the window is non-empty. The asymmetry is deliberate:
  the plan's replace-block for this tick matched the current CFB code exactly, so there was no reason
  to move code the plan did not ask to move. Flagged here for the reviewer.
- **Stricter than the plan (2 extra tests):** a `NaT` kickoff is dropped rather than snapshotted at
  an unknown distance from kickoff, and the lead window is asserted to be configurable and to default
  to 48 h. One test pins that the window filters on the *upper* bound only, so already-played games
  still reach `record_game_predictions` (which rejects them itself).

## Task 7 — `0f8df93`: `GET /api/kalshi-feed`

### RED

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_api.py
3 failed, 2.41s     # 404: the route does not exist yet
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_api.py
3 passed, 2.80s
```

- `{"sport": "cfb", "generated_at", "lead_hours", "games", "calibration"}` — matches
  `tradehub/sports/feed.py::parse_feed` exactly, ISO 8601 with `+00:00`.
- One test monkeypatches `games_data.fetch_upcoming_games`, `_load_models_cached` and `requests` to
  raise, then asserts the feed still returns 200: the feed is served from the tracking database and
  never recomputes. A live forecast is a *different* number from the frozen snapshot the hub graded,
  so recomputing would swap the series out from under the calibration check with no error anywhere.

## Full suite

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q
192 passed, 40 warnings in 10.06s      # 170 baseline + 22 new
```

## Lint

```text
uvx ruff@latest check --isolated --select E4,E7,E9,F,I <changed files>
```

Three findings, all in `src/cfb_predictor/api/routes.py` (`I001` import-block order, `F401` unused
`..odds.value_bets` import, `F841` unused `except ... as e` local) and all three present at
`origin/main`; left alone because this branch must not mix a lint sweep into a behaviour change. No
finding in any file or test added by this branch.

## Local smoke (Task 8 step 2)

Run with `CFBD_API_KEY`/`SPORTSBOOK_API_KEY` sourced from the main checkout's `.env` (read only; not
edited, not copied into the worktree) and a fresh worktree-local `data/tracking.db`. The worktree's
gitignored `data/cache/games/*.parquet` was copied from the main checkout so the smoke needed no live
CFBD call for completed seasons.

At the plan's default 48 h window the feed is **empty**, and that is correct:

```text
2026 9 26 22:02Z
fetch_upcoming_games(2026, 5) -> 59 rows, earliest kickoff 2026-10-02T00:00:00+00:00  (121 h out)
PUBLIC_MODE=false ... /tmp/smoke_feed.py cfb_predictor
2026 5 feed games: 0 lead: 48.0        # and still 0 at SNAPSHOT_LEAD_HOURS=96
[]                                    # horizon 2026-09-30 22:02Z, first kickoff 2026-10-02
```

CFB has no midweek games: week 5 is 2–3 October, so nothing is within 48 h (or 96 h) of
2026-09-26 22:02Z. Widening the window to 7 days to exercise the full path end to end:

```text
PUBLIC_MODE=false SNAPSHOT_LEAD_HOURS=168 ... /tmp/smoke_feed.py cfb_predictor
2026 5 feed games: 42 lead: 168.0
[
 {
  "game_id": "401871049",
  "season": 2026,
  "week": 5,
  "home": "New Mexico State",
  "away": "Western Kentucky",
  "start_utc": "2026-10-02T00:00:00+00:00",
  "p_home": 0.6314425631817645,
  "margin_mu": 5.701752886736627,
  "sigma": 16.98585958519707,
  "total_mu": 55.5247917175293,
  "total_sigma": 15.998238941965075,
  "home_spread_line": null,
  "total_line": null,
  "model_version": "ridge@2026-09-05T19:08:40.988453+00:00",
  "snapshotted_at": "2026-09-26T22:02:25.397472+00:00",
  "backfilled": false
 }
]
null sigma: 0 null model_version: 0 not pregame: 0
calibration n: {'winner': 0, 'spread': 0, 'total': 0}
```

`home_spread_line`/`total_line` are `null` as the plan predicted: the shared sportsbook key is out of
quota, so CFBD's line-less schedule has nothing to match. The distribution columns the hub prices
from (`margin_mu`, `sigma`, `total_mu`, `total_sigma`) are all present, so the winner market is
usable; CFB spread/total calibration stays empty until the odds key is back. Calibration `n` is 0
only because the database is fresh: buckets are built from *graded* snapshots and nothing has
finished in it.

## How the RED evidence was captured

Task 5 was committed before Tasks 6 and 7, so its RED output above was re-captured afterwards by
restoring only the task's **source** file from its parent commit and running the task's new test file
against it, then restoring (`git checkout 77d6ce6^ -- src/...` → run → `git checkout HEAD --
src/...`). No test file was modified and the worktree was verified clean afterwards. Tasks 6 and 7's
RED counts are as observed at the time of writing.

## Kevin's checklist (not the agent's)

1. Review and merge this PR. Nothing here is merged by the agent.
2. Deploy through the repo workflow's `vps` job to the VPS (`cfb.<domain>`). The first container start
   runs `_connect()`, which `ALTER TABLE`s the new distribution columns into the **persistent**
   tracking database — verify with `/api/track-record` returning normally afterwards.
3. `curl -s https://<cfb host>/api/kalshi-feed | head -c 400` must return JSON with `"sport": "cfb"`
   and `"lead_hours": 48`.
4. If CFB spread/total markets are wanted from the hub, restore the sportsbook key's quota.
   Without it `home_spread_line`/`total_line` stay `null` and the hub will not promote those markets
   to candidates until the calibration buckets fill.
5. `SNAPSHOT_LEAD_HOURS` is optional; set it in the container environment only if 48 h is wrong for
   CFB. The value the feed reports in `lead_hours` is the one the snapshot window actually used.
6. Leave the tracker running at least 48 h before the hub's first live CFB run (hub plan step 7b),
   so the feed has games with `snapshotted_at < start_utc` to serve.
