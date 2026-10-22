# CFB calibration audit (C4)

**Research only**: reads `src/cfb_predictor/` and cached data, writes nothing under `src/`,
changes no model output. Real numbers require CollegeFootballData (CFBD) API access (free tier:
1,000 calls/month; this environment lacks an API key). This doc documents the research framework
and methodology from code review.

## Model: walk-forward margin evaluation (`src/cfb_predictor/evaluate/walk_forward.py`)

The CFB walk-forward validates two candidates — "elo" and "xgb_ranker" — on held-out seasons,
predicting game margins (continuous) and derived moneyline probabilities.

### Key functions

- **`prepare_folds(games_df, fbs_teams, min_train_seasons, blocks, aux)`**: builds the full
  feature frame once (no lookahead — every feature is shift(1)/expanding past-only aggregation),
  then slices by season so `evaluate_candidate` can be called repeatedly without redoing feature
  engineering. Optional `fbs_teams` mapping excludes FCS-opponent games from training.

- **`_predict_margins(candidate, train_df, val_df, feature_cols)`**: predicts margin (home_score -
  away_score) and returns `(preds, sigma)` where sigma is the "honest" residual SD from inner
  holdout (not training-on-those-residuals).

- **`margin_to_probabilities(m, sigma, spread_line=None, total_line=None, predicted_total=None,
  total_sigma=None)`**: converts margin prediction to home_win_prob, home_cover_prob, over_prob.
  spread_line and total_line follow nflverse/convention: positive = home favored; cover when
  margin > spread_line.

- **`evaluate_candidate(folds, candidate)`**: one row per fold: val_season, n_races (games), log_loss
  / brier on the home win market (P(home win) vs actual result).

### Markets evaluated

1. **Moneyline (home win)**: predicted home_win_prob vs actual home win. Reliability slope/intercept,
   bias (mean pred prob − actual rate), sigma coverage not directly applicable (binary), regression-to-mean
   slope of actual on predicted.

2. **Spread cover**: home_cover_prob vs actual cover (margin > spread_line). Bias by favourite size
   (|spread_line| >= 7, >= 14, etc.). Sigma coverage of the margin prediction (|actual − pred| <= k*sigma).

3. **Total points**: predicted total + total_sigma → over_under prob. Bias (mean pred total − actual total),
   sigma coverage (% within ±1σ, ±2σ, ideal 68%/95%), regression-to-mean slope of actual total on
   predicted total.

### Paired-bootstrap framework

All headline metrics (bias, slope, coverage deviation from nominal) get 95% percentile bootstrap CIs
over resampled games (same games, resample-with-replacement). A fix is proposed only where the CI
excludes the ideal value (zero bias, slope=1, coverage at nominal rate).

## What I can determine from code review (no CFBD API key needed)

From reviewing `src/cfb_predictor/evaluate/walk_forward.py`, `src/cfb_predictor/evaluate/calibration.py`,
and `src/cfb_predictor/features/build.py`:

- The CFB model uses Elo-based margins and XGBoost ranker, similar in structure to NFL but with
  different data source (CFBD vs nflverse).
- The `sigma_from_residuals` function computes `np.std(residuals, ddof=1)` — the honest sigma from
  inner holdout residuals. As with NFL, this tends to be conservative (sigma wider than actual error
  SD), making sigma coverage ~100% rather than 68%/95%.
- The `evaluate_candidate` returns log_loss and brier per fold for the home-win market.
- `QUANTILE_MARKETS` and `LINE_COLUMN` constants indicate the model evaluates spread and total
  markets via `margin_to_probabilities` with real spread_line and total_line from the games data.
- The build frame includes features like power ratings, rolling form, rest days, conference info,
  and FBS team filtering — similar to NFL's feature pipeline.

### Data access needed for real numbers

- CollegeFootballData API key (free tier: 1,000 calls/month; this environment lacks a key).
- Alternatively, cached parquet files under `data/cache/cfbd/` (present in the repo but gitignored;
  not available in the worktree).
- The `load_training_data` function in `cfb_predictor/data/games.py` fetches per-season schedules
  and results from CFBD (one call per season under the free-tier cap).

### Plain reading

- **No problem assessed** from code review alone — the CFB walk-forward framework is structured
  similarly to NFL's and produces per-fold log_loss/brier metrics. Actual calibration numbers
  (reliability slope, bias, sigma coverage) require CFBD data access.
- **Fix warranted** if paired-bootstrap CI for any key metric (bias ≠ 0, slope ≠ 1, coverage ≠ nominal)
  excludes the ideal value. Typical CFB findings (from the existing codebase's audit notes in
  `docs/superpowers/`) note that big favourites win by slightly less than predicted (−0.83 for 12+ pt
  favourites), suggesting mild over-confidence on large spreads — a fix (e.g. shrinkage) would be
  warranted only if the paired-bootstrap CI for the slope excludes 1.0.
- **No fix shipped**: this is research only; no serving change is proposed without a paired-bootstrap
  CI excluding the ideal on re-tested candidates.

### Next steps (with CFBD API access)

1. Obtain a CFBD API key and configure it in the environment.
2. Run `prepare_folds` + `evaluate_candidate` for held-out seasons (e.g., 2019-2025, n ≈ 8,388
   games as in the NBA audit write-up).
3. Compute per-market: reliability slope/intercept per market, bias by situation (home/away, early
   season, favourite size), sigma coverage (% within ±1σ, ±2σ vs nominal 68%/95%), regression-to-mean
   slope for margin and total.
4. Paired-bootstrap 95% CIs for each headline metric.
5. If any CI excludes the ideal value → propose a fix (e.g. shrinkage on big favourites, re-calibration
   of totals sigma).
6. If all CIs include ideal values → report "no problem" across the board.