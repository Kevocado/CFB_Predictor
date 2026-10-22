# CFB calibration audit (public snapshot weeks 1-6)

Generated 2026-10-22. Ridge margin model, weeks 1-6 (2026 season), 448 completed games. Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00 / intercept 0.00; bias 0; sigma coverage 0.683 (1σ) / 0.954 (2σ);
regression-to-the-mean slope 1.00. Max calibration gap <0.03 reads as honest.

## Results

| Market | Metric | Value | 95% CI | Reading |
|--------|--------|-------|--------|---------|
| Moneyline (home win) | Reliability slope | 0.998 | [0.811, 1.102] | no problem |
| Moneyline (home win) | Reliability intercept | 0.046 | — | ideal 0.00 |
| Moneyline (home win) | Max calibration gap (pts) | 0.150 | — | <0.03 honest |
| Moneyline (home win) | Mean predicted prob − actual rate | -0.042 | [-0.076, -0.006] | problem |
| Moneyline | Decile bins (min_n=5) | [0.07,0.14,n=7.0] | [0.16,0.13,n=15.0] | [0.25,0.28,n=18.0] | [0.35,0.50,n=28.0] | [0.45,0.44,n=48.0] | [0.55,0.64,n=36.0] | [0.65,0.67,n=48.0] | [0.75,0.85,n=75.0] | [0.85,0.85,n=84.0] | [0.95,0.99,n=89.0] | — | |
| Margin | Mean predicted − actual (pts) | -5.625 | [-7.426, -3.833] | problem |
| Margin | Within 1 sigma | 0.618 | [0.574, 0.663] | expected 0.683 |
| Margin | Within 2 sigma | 0.911 | [0.884, 0.935] | expected 0.954 |
| Margin | Regression-to-mean slope | 1.096 | [0.975, 1.218] | no problem |
| Total | Mean predicted − actual (pts) | -2.258 | [-3.767, -0.747] | problem |
| Total | Within 1 sigma | 0.676 | [0.627, 0.719] | expected 0.683 |
| Total | Within 2 sigma | 0.955 | [0.935, 0.973] | expected 0.954 |
| Spread cover | — | cover probs None in snapshot | — | skipped |
| Moneyline | Bias home favoured | -0.045 | -0.085..-0.007 | |
| Moneyline | Bias away favoured | -0.037 | -0.125..+0.048 | |
| Moneyline | Bias |margin|>=14 | -0.019 | -0.056..+0.021 | |
| Margin | Bias |margin|>=14 | -7.545 | -10.158..-4.983 | |

## Interpretation
- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.
- Fix proposed only where paired-bootstrap CI excludes the ideal value (bias≠0, slope≠1, coverage≠nominal).
- Data source: public_snapshot.json weeks 1-6 (2026 season), 437 completed games with model predictions.
- No CFBD API key needed — used public snapshot data with model predictions and actual outcomes.
- No fix shipped here: research only, no serving change proposed without paired-bootstrap CI excluding ideal on re-tested candidates.
