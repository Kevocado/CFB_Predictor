# CFBD advanced-stats pull → `epa` block evaluation

## The pull

| Item | Value |
|---|---|
| Years pulled | **2014–2025** (12 seasons) |
| **API calls used** | **12** (exactly one per year) |
| Budget | 40 → 12 used, 28 unspent |
| Files | `data/cfbd/advanced_<year>.json`, one per year, 12 files |
| Endpoint | `GET /stats/game/advanced` |
| Key source | `CFBD_API_KEY` from the environment; never printed, echoed or written to disk |

The pull script is now part of the repo:

```bash
python -m cfb_predictor.data.cfbd_advanced --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
# -> requested years: [2014 ... 2025]
# -> API calls made this run: 12
# -> total API calls for the whole pull: 12
```

It is resumable (a year already on disk is skipped, not re-fetched), budgeted (raises
`BudgetExceeded` rather than overspending), and writes atomically so an interrupted write never
leaves a file the next pull would mistake for "done".

`to_team_game_frame()` turns the raw rows into the efficiency frame the `epa` block reads. One
detail that has to be handled: **the season is not the leading digits of the CFBD game id**. The
2014 file's first id is `400547640` and the 2025 file's is `401752665`, so `game_id[:4]` is wrong.
The frame now carries `season` and `week` straight through from the CFBD row.

## `epa` block: evaluated

`prepare_folds` did not forward `blocks`/`aux` to `build_training_frame`, so the epa block could be
wired in `_assemble` and still never reachable from the walk-forward loop. That is now fixed, and
the block evaluates:

```bash
python -m cfb_predictor.tools.block_eval --blocks epa priors --start-year 2014 --end-year 2025
# raw output is committed at output/cfb_block_eval.txt
```

| Block | N | MAE Δ | MAE 95% CI | Brier Δ | Brier 95% CI | Gap Δ 95% CI | Clears |
|---|---|---|---|---|---|---|---|
| epa | 7560 | +0.00420 | [−0.05329, +0.06109] | −0.00013 | [−0.00109, +0.00084] | [−0.01573, +0.00663] | **No** |

Same walk-forward protocol as the NFL evaluation: seasons are folds, each fold trains on every
season strictly before it, and base / with-block runs see identical held-out games. `Δ = base −
block`, so positive is better. EPA's point estimates are flat-to-slightly-positive and every
interval straddles zero — no evidence it helps, no evidence it hurts. It does **not** clear.

## `priors` block: wired, evaluated, **left OFF**

`features/priors.add_priors` is now wired into `_assemble` like `epa`, adding `home_prior`, `away_prior`,
`prior_diff`. A team's season-s prior is its Elo going into its last season-(s-1) game (omitting that one game's update),
regressed 40% toward the mean of its season-(s-1) conference over the complete prior-season field
(`preseason_prior`), so a served prior does not depend on who has played in season s yet; a team with no
season-(s-1) history gets its own conference's mean; the first season in the data is neutral (1500). It needs no `Aux` input, so training
(`train_all`) and serving build it with no extra data. No new CFBD call: it is derived from the games already cached.
Serving also now opens a new season for a game dated July or later in a year the data has not reached yet.

Same walk-forward protocol and games as `epa` (N=7560 held-out games, ridge, 2014-2025 data, seasons 2016-2025 held out).
Raw output: `output/cfb_block_eval.txt`. `epa` reproduces the numbers above exactly.

| Block | MAE d | MAE 95% CI | Brier d | Brier 95% CI | AUC d (block - base) | AUC 95% CI | Gap base -> block | Gap d 95% CI |
|---|---|---|---|---|---|---|---|---|
| priors | -0.00542 | [-0.01595, +0.00625] | -0.00003 | [-0.00022, +0.00015] | -0.00027 | [-0.00075, +0.00018] | 0.02568 -> 0.02950 | [-0.01162, +0.00341] |

Verdict: **OFF**. The AUC interval does not exclude a decline (lower bound -0.00075) and the calibration gap
widens (0.0257 -> 0.0295), so `DEFAULT_BLOCKS` stays `()`. The prior is mostly what the Elo
already carries; the fixed-within-season prior adds nothing a linear model can use. The returning-production and
recruiting extension (Task 12 step 4) would cost new CFBD calls and was not run.

## Two corrected claims from the earlier draft of this doc

An earlier version of this file said the CFB pipeline "does not accept auxiliary inputs", so both
blocks were unevaluable. That was wrong, and the correction matters:

- **`epa` does accept aux.** `build_training_frame(..., blocks, aux)` passes `aux.efficiency` to
  `epa.add_epa_features`, and `_assemble` already refused to default it to zeros. The only gap was
  `prepare_folds` not forwarding it, now fixed.
- **`priors` was not wired** (now it is; see above).

## `DEFAULT_BLOCKS`: unchanged

`DEFAULT_BLOCKS` is `()`. Nothing in this evaluation supports changing it: `epa` does not clear,
and `priors` is measured and does not clear.

## Files

`data/cfbd/advanced_2014.json` … `advanced_2025.json` (12 files, ~48 MB total).

## CFB tests

```bash
PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m pytest tests/ -q
```
