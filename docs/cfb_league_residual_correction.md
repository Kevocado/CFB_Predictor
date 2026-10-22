# CFB trailing league-residual correction — ship/no-ship result

**Date:** 2026-10-22  
**Gate verdict:** NO SHIP — flag stays OFF (docs + code behind flag).

---

## One-line conclusion

**The trailing league-residual correction does not improve CFB margin or total MAE on the 2026 snapshot (weeks 4-7, 250 games); flag stays OFF. Candidate to re-run at week 12.**

---

## Context

CFB Ridge margin model under-predicts this season:
- Margin bias: **-5.6 pts** (CI [-7.4, -3.8])
- Total bias: **-2.26 pts** (CI [-3.77, -0.75])
- Audit: `docs/cfb_calibration_audit.md` (weeks 1-6, 448 games)

Port of the NBA-style causal trailing league-residual correction:
`correction = mean(actual - predicted)` over the last N played games LEAGUE-WIDE
(not per team), capped, added to the raw prediction. Strictly causal: predicting
week k uses only weeks < k.

---

## Parameters (chosen and justified)

| Param | Value | Justification |
|-------|-------|---------------|
| Window N | 300 games | Matches the NBA window (~300 games). At ~75 CFB games/week this spans ~4 weeks — enough samples for a stable mean (SE ~0.5-1 pt) while staying responsive to regime changes within a season. |
| Cap (margin) | ±8 pts | Observed margin bias is -5.6, so the cap must allow at least 6; ±8 captures the full correction while preventing extreme over-correction on thin early-season samples. |
| Cap (total) | ±4 pts | Matches the NBA total cap; observed total bias is -2.26, so ±4 is ample and prevents over-correction on noisy totals. |

---

## Ship test (walk-forward, weeks 4-6)

Warm-up week 4: predict week k using residuals from weeks < k (strictly causal).
Paired bootstrap resamples GAMES (N=2000) for the CI of (raw_abs_error - corrected_abs_error).

### Per-week raw vs corrected MAE

| Week | n | prior games | margin corr | total corr | margin MAE raw | margin MAE corr | total MAE raw | total MAE corr |
|------|---|-------------|-------------|------------|----------------|-----------------|---------------|----------------|
| 4 | 71 | 260 | +8.000 | +4.000 | 15.596 | 16.122 | 11.873 | 12.436 |
| 5 | 59 | 331 | +6.895 | +3.289 | 14.738 | 15.021 | 13.815 | 14.010 |
| 6 | 58 | 390 | +4.734 | +2.520 | 12.285 | 12.061 | 13.473 | 14.150 |
| 7 | 62 | 448 | +3.281 | +1.159 | 11.432 | 11.344 | 12.890 | 13.035 |

### Pooled (250 games, weeks 4-7)

| Metric | raw MAE | corr MAE | delta (raw-corr) | CI (paired games bootstrap) | excludes zero? |
|--------|---------|----------|------------------|----------------------------|----------------|
| Margin | 13.593 | 13.735 | -0.142 | (-0.869, +0.588) | **No** |
| Total | 12.955 | 13.353 | -0.399 | (-0.762, -0.033) | Yes (but corrected is **worse**) |

### Buckets (margin)

| Bucket | n | raw MAE | corr MAE | delta |
|--------|---|---------|----------|-------|
| home_fav (small, \|margin\|<14) | 60 | 10.288 | 14.609 | **-4.320 (worse)** |
| home_fav (big, \|margin\|>=14) | 85 | 17.324 | 14.787 | +2.537 |
| away_fav (small, \|margin\|<14) | 61 | 7.999 | 5.860 | +2.139 |
| away_fav (big, \|margin\|>=14) | 44 | 18.644 | 21.428 | **-2.784 (worse)** |

---

## Ship gate

Ship (flag ON) only if:
1. Paired-bootstrap CI of (raw_abs - corrected_abs) excludes zero for the metric, AND
2. No bucket (home/away favourite, \|margin\|>=14) is worse.

**Result: NO SHIP.**
- Margin CI includes zero (-1.151, +0.709).
- Total CI excludes zero but is negative (corrected is worse).
- Two buckets (home_fav small, away_fav big) get worse.

The correction does not help on this thin sample. **Flag stays OFF.**

---

## Code

- Flag: `config.CFB_LEAGUE_RESIDUAL_CORRECTION` (env `CFB_LEAGUE_RESIDUAL_CORRECTION`, default `false`).
- Correction function: `game_outcome.league_residual_correction(residuals, window, cap)`.
- Constants: `LEAGUE_RESIDUAL_WINDOW=300`, `LEAGUE_RESIDUAL_CAP_MARGIN=8.0`, `LEAGUE_RESIDUAL_CAP_TOTAL=4.0`.
- Serving wiring: `_predict_game_from_models` applies correction when flag is ON (cached per week).
- Ship test: `scripts/cfb_league_residual_ship_test.py`.

**`src/cfb_predictor/models/league_residual_correction` is research code behind a flag; not active in serving.**

---

## Follow-up

Candidate to re-run at week 12 with more completed games. The correction is
implemented and tested behind the flag; flipping the flag to `true` requires
re-running the ship test and confirming the CI excludes zero with no bucket worse.
