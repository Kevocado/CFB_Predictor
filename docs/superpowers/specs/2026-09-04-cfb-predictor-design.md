# CFB Predictor: Design Spec

**Date:** 2026-09-04
**Status:** approved, pending implementation plan

## Purpose

A self-hosted FBS college football prediction dashboard: pre-kickoff win/
spread/total probabilities for every game on the weekly slate, plus
player-level touchdown and yardage predictions for skill-position
players — served through a FastAPI backend and a React dashboard, with
every prediction tracked against what actually happens. Fourth sibling
project alongside `PL_Predictor`, `F1_Predictor`, and `NFL_Predictor`,
reusing `NFL_Predictor`'s architecture near-verbatim (it is the closest
sibling by a wide margin: same sport, same two-team/high-count/bursty-
scoring shape, same walk-forward-raced game-outcome model, same
snapshot-then-reconcile tracking, same Shin-de-vig value-bet comparison).

## What actually changes vs. NFL_Predictor (and what doesn't)

This is the central design question this spec answers, since the whole
point of a fourth sibling is reuse. Verdict: **the feature/model/
tracking/API/frontend architecture transfers unchanged; only the data
layer and a handful of CFB-shaped features are new.**

**Transfers unchanged (copy the file, at most rename `nfl_predictor` →
`cfb_predictor` in imports):**
- `features/build.py`'s single-entry-point discipline, `rolling_form.py`,
  `rest_days.py` (bye weeks are rarer in CFB but the shift(1)/no-lookahead
  logic is identical), `power_ratings.py`'s margin-multiplier Elo update
  (already generic over any two-team margin).
- `models/game_outcome.py`'s three-candidate race (elo / ridge / xgb),
  `margin_to_probabilities`, `evaluate/walk_forward.py`'s fold-by-season
  harness — all sport-agnostic already; NFL_Predictor's own code never
  references anything NFL-specific in these files.
- `models/player_props.py`'s anytime-TD classifier + yardage-regressor
  shape, `features/player_usage.py`'s rolling-usage pattern.
- `tracking/store.py` verbatim (SQLite snapshot-then-reconcile has zero
  sport-specific logic) — including the just-landed fixes from
  NFL_Predictor's final review (`contextlib.closing` connection handling).
- `odds/value_bets.py` verbatim (Shin de-vig math is market-agnostic).
- `api/main.py`, `api/routes.py`'s route shape, `api/schemas.py` — same
  endpoints, same lifespan/background-tracking-tick pattern (including
  NFL_Predictor's hardened per-game try/except and PUBLIC_MODE retrain
  gate, carried over from day one instead of re-discovered).
- The entire `frontend/` app — same tab shell, same three pages, same
  `api/client.ts` shape (including the corrected same-origin `/api`
  `BASE_URL` and Vite dev proxy, carried over from day one).
- `Dockerfile` (including the `$PORT` fix), `.dockerignore`.

**Genuinely new (the actual work of this plan):**
1. **Data layer.** `nfl_data_py` (free, keyless, unlimited) has no CFB
   equivalent at that level of convenience. The closest real source is
   [CollegeFootballData.com](https://collegefootballdata.com) (CFBD) via
   its official `cfbd` Python package — free but API-**keyed** (a free
   key requested once at
   [collegefootballdata.com/key](https://collegefootballdata.com/key),
   sent as a Bearer token) and **rate-capped at 1,000 calls/month on the
   free tier** ([collegefootballdata.com/api-tiers](https://collegefootballdata.com/api-tiers)).
   This is the one real architectural constraint NFL_Predictor never had
   to design around: the data layer must be aggressively cache-first and
   batch its calls (one call per season for schedules/team-game-stats,
   not one call per team or per week), because a naive per-request-shaped
   fetch pattern would blow the monthly cap in days. See Data Layer below
   for the exact call budget.
2. **Team universe size and structure.** ~134 FBS teams across 10
   conferences + independents (vs. NFL's 32 teams / 2 conferences), and
   FBS teams occasionally play FCS opponents (usually lopsided). Two
   feature-layer decisions follow: use a `conference_game` boolean flag
   in place of NFL's `div_game`, and exclude FCS-opponent games from
   model training targets (they're a different competitive tier and would
   distort power ratings) while still counting them for the FBS team's
   rest-days/rolling-form bookkeeping.
3. **Odds coverage is unverified and likely sparser.** The Odds API
   supports CFB under sport key `americanfootball_ncaaf` (confirmed
   supported market types: h2h/spreads/totals), but nowhere near every
   FBS matchup gets a line, especially outside Power-conference games.
   Same non-blocking pattern as NFL_Predictor's original spec: value-bet
   detection degrades gracefully (no recommendation) for any game with no
   quoted market, checked empirically during implementation, not designed
   around in advance.
4. **New API key required.** Unlike NFL_Predictor (zero new keys beyond
   the already-shared Odds API key), this project needs one new free
   `CFBD_API_KEY`. No new *paid* keys — the free tier is the plan.

**Deliberately NOT changing the scope to fit CFB's real complexity:**
top-25 rankings (AP/Coaches/CFP poll) are a real, CFB-specific signal
with no NFL equivalent, but are explicitly **out of scope for v1** (see
below) to keep this plan's shape matched to NFL_Predictor's rather than
growing it — a natural v1.1 addition once the core race is proven, exactly
the same "prove the core model first" posture NFL_Predictor's own spec
took toward its live in-game engine.

## Scope (v1)

**In scope:**
- Game-level predictions for every FBS-vs-FBS regular-season (and, once
  reached, postseason/bowl) game: moneyline win probability, point-spread
  cover probability, over/under total-points probability.
- Player props: anytime-touchdown-scorer probability, QB passing
  yards/TD probability, RB rushing yards/TD probability, WR/TE receiving
  yards/TD probability — FBS skill players only.
- Value-bet detection: model probability vs. The Odds API's de-vigged
  CFB lines, same one-recommendation-per-market-never-a-parlay rule.
- Honest tracking: every prediction snapshotted before kickoff, reconciled
  against results as they land.
- React frontend: weekly-slate view, player props view, track record —
  same three tabs as NFL_Predictor.
- Docker → Render deploy, plus a fourth card on `predictor-hub/index.html`.

**Explicitly out of scope for v1:**
- AP/Coaches/CFP poll ranking as a feature (real signal, deliberately
  deferred — see above).
- FCS teams as first-class predictable entities (their games against FBS
  opponents are excluded from training; FCS-vs-FCS is out of scope
  entirely — CFBD's FCS data coverage is also considerably thinner).
- Live in-game win-probability engine and playoff/CFP bracket Monte Carlo
  projection — same deferral NFL_Predictor made for its own live engine
  and championship projection, for the same reason (prove the core
  game/player models first).
- Any prop market beyond the four listed.

## Project layout

New sibling repo at `Documents/Projects/CFB_Predictor` (top-level,
alongside `NFL_Predictor`/`F1_Predictor`/`PL_Predictor`). Own git repo
(already initialized), own Python venv.

```
src/cfb_predictor/
  config.py                 paths, env loading, shared constants
                             (adds CFBD_API_KEY; everything else matches
                             nfl_predictor/config.py's names/shapes)
  data/                      cfbd_client.py (schedules/team-game-stats/
                             player weekly stats via the cfbd package,
                             replaces nfl_data_py), odds_api.py (near-
                             identical to NFL's, sport key swapped)
  features/                  build.py, power_ratings.py, rolling_form.py,
                             rest_days.py, player_usage.py
                             (all near-verbatim ports from nfl_predictor)
  models/                    game_outcome.py, player_props.py, manifest.py
                             (near-verbatim ports)
  evaluate/                  walk_forward.py (verbatim port)
  tracking/                  store.py (verbatim port)
  odds/                      value_bets.py (verbatim port)
  api/                       main.py, routes.py, schemas.py
                             (near-verbatim ports)
frontend/src/                (near-verbatim port of nfl_predictor's
                             frontend — GamesPage/PlayerPropsPage/
                             TrackRecordPage, tab shell, api/client.ts)
tests/
docs/
  superpowers/specs/         this file
  superpowers/plans/         implementation plan(s)
Dockerfile
pyproject.toml
```

## Data layer

- **Primary source: CFBD (`cfbd` Python package)**, hitting
  `api.collegefootballdata.com` with a Bearer-token free API key
  (`CFBD_API_KEY` env var, requested once at
  [collegefootballdata.com/key](https://collegefootballdata.com/key)).
  Provides schedules/games, team season/game stats, player weekly usage
  stats, and rosters — CFBD is this project's `nfl_data_py` equivalent,
  just keyed and rate-capped instead of free-and-unlimited.
- **Hard constraint: 1,000 calls/month on the free tier.** The data
  modules MUST fetch per-season (or per-season-and-week where the
  endpoint requires it), never per-team or per-game, and cache
  aggressively to `data/cache/` exactly like NFL_Predictor's
  cache-or-fetch pattern — but here the cache is load-bearing for staying
  under the cap, not just a latency optimization. A back-of-envelope
  budget for an 8-season training window refreshed weekly in-season:
  ~8 calls for historical season game data (already cached after first
  fetch, near-zero ongoing cost) + ~4 calls/week in-season for the
  current season's games/team-stats/player-stats/rosters ≈ well under
  1,000/month even with headroom for manual re-fetches during
  development. This budget is a design constraint every data-module task
  must respect, not a one-time check.
- **The Odds API** — same account/key already used by `PL_Predictor` and
  `NFL_Predictor`. Sport key `americanfootball_ncaaf`. Whether the
  current plan tier carries CFB player-prop markets, and how many FBS
  games get a line at all outside Power-conference matchups, is an
  empirical check during implementation (non-blocking, same posture
  NFL_Predictor's spec took) — if coverage is sparse or absent, player
  props ship model-only (no value-bet comparison) and game-level value
  bets degrade gracefully per-game (no recommendation where there's no
  quoted market), never a hard failure.
- No new *paid* keys. One new *free* key (`CFBD_API_KEY`).

## Features

**Team-level:** offense/defense power ratings (updated per week, same
Elo-style margin-multiplier update as NFL_Predictor — verified generic,
no NFL-specific assumption in that code), rolling scoring/yardage form,
rest days since last game, home-field indicator, **`conference_game`**
flag (replaces NFL's `div_game` — true when both teams share a
conference; false for inter-conference and FBS-vs-independent games).

**Player-level:** usage share, rolling yards-per-game — identical shape
to NFL_Predictor's `player_usage.py`.

**FCS-opponent handling:** a game where either team is not FBS is
excluded from `build_training_frame`'s target rows (it's not a fair
signal for FBS-vs-FBS power ratings), but still counted in the FBS
team's `rest_days`/`rolling_form` bookkeeping (the team genuinely played
and rested on that date) — same "features computed over full history,
model trained only on the relevant subset" pattern already implicit in
how NFL_Predictor's `build.py` separates feature computation from
training-row selection.

All consumers go through one `features/build.py` entry point, same
discipline as NFL_Predictor.

## Models

Identical candidate race and selection discipline to NFL_Predictor:
1. Elo / power-rating baseline.
2. Ridge margin-of-victory regression.
3. XGBoost regression on the full feature set.

Walk-forward validated on held-out later seasons, log-loss selection,
`manifest.py` records the winner — verbatim port of NFL_Predictor's
`evaluate/walk_forward.py` and `models/manifest.py` (both already
sport-agnostic; the only NFL-specific surface in the entire modeling
stack was the `div_game`/`conference_game` feature name, handled above).

**Player props:** anytime-TD binary classifier, passing/rushing/
receiving yards regressions — same shape as NFL_Predictor.

## Tracking and value bets

Verbatim port of `tracking/store.py` (immutable snapshot-before-kickoff
→ reconcile-after-result SQLite, including the connection-handling fix
from NFL_Predictor's final review) and `odds/value_bets.py` (Shin de-vig,
at most one qualified single per market, never a parlay).

## API

Same route shape as NFL_Predictor, same PUBLIC_MODE-gated retrain:
- `GET /api/games?season=2026&week=1` — weekly slate
- `GET /api/games/{season}/{week}/{game_id}/prediction` — win/spread/
  total probability table for a game
- `GET /api/players/{season}/{week}/props` — player prop predictions
- `GET /api/track-record` — honest calibration summary
- `POST /api/retrain` — retrain all models (403 when `PUBLIC_MODE=true`)

## Frontend

Near-verbatim port of NFL_Predictor's React 19 + TypeScript + Vite app:
Games / Player Props / Track Record tabs, same `api/client.ts` shape
(same-origin `/api` BASE_URL with a Vite dev proxy — carried over
correctly from day one instead of re-discovered by a future final
review).

## Deploy and hub integration

Dockerfile → Render, same pattern as the other three projects (including
the `$PORT`-binding fix carried over from day one). Once live, add a
fourth card to `predictor-hub/index.html` (🎓 or similar icon, "Live"
badge) alongside the existing PL/F1/NFL cards.

## Testing

Pytest suite mirroring NFL_Predictor's coverage shape: data module tests
(cache-or-fetch correctness, call-budget-conscious mocking of the `cfbd`
client so tests never make real network calls against the rate cap),
feature tests (no lookahead, `conference_game` correctness,
FCS-opponent exclusion-from-training-but-not-from-rest-days behavior),
model tests (walk-forward harness correctness — reused verbatim),
API tests, tracking tests (reused verbatim).

## Open items to confirm during implementation

- Whether the existing Odds API plan includes CFB player-prop markets,
  and roughly what fraction of a typical week's FBS slate has ANY quoted
  h2h/spreads/totals line (affects how much value-bet coverage is
  realistic to expect, not a blocking design decision).
- CFBD's exact endpoint shapes for team-game stats and player weekly
  stats (field names) — confirmed against the real API during the data-
  layer tasks, same "verify against real data" posture NFL_Predictor
  used for its own `KEEP_COLUMNS` lists.
- Whether CFBD player ids remain stable across a transfer-portal move
  (a player changing teams mid-career) — if not, rolling player-level
  form should reset at the team boundary rather than silently carrying a
  new team's jersey under an old team's rolling averages. Checked
  empirically, not designed around speculatively.
