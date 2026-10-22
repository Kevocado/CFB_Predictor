#!/usr/bin/env python3
"""cfb_league_residual_ship_test.py — ship/no-ship gate for the trailing
league-residual correction on CFB margin and total.

Walks forward week by week through the 2026 snapshot from a warm-up week:
for each test week k, the correction uses only residuals from weeks < k
(strictly causal). Compares corrected vs raw absolute error (margin and
total separately) with a paired bootstrap that resamples GAMES for the CI
of (raw_abs_error - corrected_abs_error).

Ship (flag ON) only if that CI excludes zero for the metric AND no bucket
(home/away favourite, |margin|>=14) is worse. Otherwise flag stays off.

Run:
    python scripts/cfb_league_residual_ship_test.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from cfb_predictor.models import game_outcome

SNAPSHOT = "data/public_snapshot.json"
WARMUP_WEEK = 4
N_BOOT = 2000
SEED = 20261022
WINDOW = game_outcome.LEAGUE_RESIDUAL_WINDOW  # 300
CAP_MARGIN = game_outcome.LEAGUE_RESIDUAL_CAP_MARGIN  # 8
CAP_TOTAL = game_outcome.LEAGUE_RESIDUAL_CAP_TOTAL  # 4


def _games_with_preds(snap, week):
    w = snap["weeks"][str(week)]
    games = w["games"]
    preds = w["predictions"]
    out = []
    for g in games:
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        p = preds.get(str(g["game_id"]))
        if not p:
            continue
        if p.get("predicted_margin") is None or p.get("predicted_total") is None:
            continue
        out.append((g, p))
    return out


def _residuals_from_weeks(snap, weeks):
    """Collect (actual_margin, pred_margin, actual_total, pred_total) for completed games."""
    rows = []
    for w in weeks:
        for g, p in _games_with_preds(snap, w):
            am = float(g["home_score"]) - float(g["away_score"])
            at = float(g["home_score"]) + float(g["away_score"])
            rows.append((am, float(p["predicted_margin"]), at, float(p["predicted_total"]), g))
    return rows


def _bucket(g):
    am = float(g["home_score"]) - float(g["away_score"])
    # home/away favourite by predicted margin sign
    pm = None
    # use predicted margin from... we'll pass separately; here use actual margin sign as proxy? No.
    # Bucket by predicted margin sign + magnitude.
    return None  # placeholder, filled below


def _bucket_of(g, pred_margin):
    if pred_margin >= 0:
        fav = "home_fav"
    else:
        fav = "away_fav"
    am = abs(float(g["home_score"]) - float(g["away_score"]))
    if am >= 14:
        mag = "big"
    else:
        mag = "small"
    return fav, mag


def main() -> int:
    snap = json.load(open(SNAPSHOT))
    # weeks with completed games
    completed_weeks = []
    for w in range(1, 17):
        gs = _games_with_preds(snap, w)
        if len(gs) >= 10:
            completed_weeks.append(w)
    print(f"Completed weeks: {completed_weeks}")

    warmup = WARMUP_WEEK
    test_weeks = [w for w in completed_weeks if w >= warmup]
    print(f"Warm-up week: {warmup}; test weeks: {test_weeks}")

    # Walk forward
    per_week = []
    all_rows = []  # (raw_abs_margin, corr_abs_margin, raw_abs_total, corr_abs_total, bucket)
    for k in test_weeks:
        prior_weeks = [w for w in completed_weeks if w < k]
        prior = _residuals_from_weeks(snap, prior_weeks)
        margin_resid = [r[0] - r[1] for r in prior]
        total_resid = [r[2] - r[3] for r in prior]
        corr_m = game_outcome.league_residual_correction(margin_resid, window=WINDOW, cap=CAP_MARGIN)
        corr_t = game_outcome.league_residual_correction(total_resid, window=WINDOW, cap=CAP_TOTAL)
        print(f"\nWeek {k}: prior games={len(prior)}  margin_corr={corr_m:+.3f}  total_corr={corr_t:+.3f}")

        week_games = _games_with_preds(snap, k)
        raw_mae_m = []; corr_mae_m = []
        raw_mae_t = []; corr_mae_t = []
        for g, p in week_games:
            am = float(g["home_score"]) - float(g["away_score"])
            at = float(g["home_score"]) + float(g["away_score"])
            pm = float(p["predicted_margin"])
            pt = float(p["predicted_total"])
            raw_ae_m = abs(am - pm)
            corr_ae_m = abs(am - (pm + corr_m))
            raw_ae_t = abs(at - pt)
            corr_ae_t = abs(at - (pt + corr_t))
            raw_mae_m.append(raw_ae_m); corr_mae_m.append(corr_ae_m)
            raw_mae_t.append(raw_ae_t); corr_mae_t.append(corr_ae_t)
            bucket = _bucket_of(g, pm)
            all_rows.append((raw_ae_m, corr_ae_m, raw_ae_t, corr_ae_t, bucket, k))

        print(f"  n={len(week_games)}  margin MAE raw={np.mean(raw_mae_m):.3f} corr={np.mean(corr_mae_m):.3f}  "
              f"total MAE raw={np.mean(raw_mae_t):.3f} corr={np.mean(corr_mae_t):.3f}")
        per_week.append((k, len(week_games), np.mean(raw_mae_m), np.mean(corr_mae_m),
                        np.mean(raw_mae_t), np.mean(corr_mae_t), corr_m, corr_t))

    # Pooled paired bootstrap over games
    raw_m = np.array([r[0] for r in all_rows])
    corr_m = np.array([r[1] for r in all_rows])
    raw_t = np.array([r[2] for r in all_rows])
    corr_t = np.array([r[3] for r in all_rows])

    def paired_ci(raw, corr, seed=SEED):
        n = len(raw)
        rng = np.random.default_rng(seed)
        deltas = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            deltas[i] = np.mean(raw[idx] - corr[idx])  # raw_abs - corrected_abs
        return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

    ci_m = paired_ci(raw_m, corr_m)
    ci_t = paired_ci(raw_t, corr_t)

    print("\n" + "=" * 70)
    print("POOLED (weeks %d-%d, n=%d games)" % (test_weeks[0], test_weeks[-1], len(all_rows)))
    print("  Margin: raw MAE=%.3f  corr MAE=%.3f  delta (raw-corr)=%.3f  CI=%s" % (
        raw_m.mean(), corr_m.mean(), (raw_m - corr_m).mean(), ci_m))
    print("  Total:  raw MAE=%.3f  corr MAE=%.3f  delta (raw-corr)=%.3f  CI=%s" % (
        raw_t.mean(), corr_t.mean(), (raw_t - corr_t).mean(), ci_t))

    # Buckets
    print("\nBuckets (margin):")
    for bucket_name in ["home_fav", "away_fav"]:
        for mag in ["small", "big"]:
            sel = [r for r in all_rows if r[4][0] == bucket_name and r[4][1] == mag]
            if not sel:
                continue
            rr = np.array([r[0] for r in sel]); cc = np.array([r[1] for r in sel])
            print(f"  {bucket_name}_{mag} (|margin|{'>=14' if mag=='big' else '<14'}): n={len(sel)} "
                  f"raw={rr.mean():.3f} corr={cc.mean():.3f} delta={ (rr-cc).mean():+.3f}")

    # Ship decision
    print("\n" + "=" * 70)
    ship_margin = ci_m[0] > 0 or ci_m[1] < 0  # CI excludes zero
    # CI excludes zero means 0 not in [lo, hi]
    ship_margin = not (ci_m[0] <= 0 <= ci_m[1])
    ship_total = not (ci_t[0] <= 0 <= ci_t[1])
    print(f"Margin CI excludes zero: {ship_margin}  {ci_m}")
    print(f"Total  CI excludes zero: {ship_total}  {ci_t}")

    # bucket worse check: corrected MAE > raw MAE for any bucket
    bucket_worse = False
    for bucket_name in ["home_fav", "away_fav"]:
        for mag in ["small", "big"]:
            sel = [r for r in all_rows if r[4][0] == bucket_name and r[4][1] == mag]
            if not sel:
                continue
            rr = np.array([r[0] for r in sel]); cc = np.array([r[1] for r in sel])
            if cc.mean() > rr.mean():
                bucket_worse = True
                print(f"  BUCKET WORSE: {bucket_name}_{mag} raw={rr.mean():.3f} corr={cc.mean():.3f}")

    ship = (ship_margin or ship_total) and not bucket_worse
    # Actually ship requires BOTH? Task: "Ship (flag ON) only if that CI excludes zero for the metric"
    # Per metric. But flag is single. Ship if at least one metric improves AND no bucket worse?
    # Let's require: ship if (margin CI excludes zero AND margin improves) or (total CI excludes zero AND total improves), and no bucket worse.
    margin_improves = ship_margin and (raw_m.mean() > corr_m.mean())
    total_improves = ship_total and (raw_t.mean() > corr_t.mean())
    ship = (margin_improves or total_improves) and not bucket_worse

    print(f"\nSHIP DECISION: {'SHIP (flag ON)' if ship else 'NO SHIP (flag stays OFF)'}")
    if not ship:
        print("  Reason: CI includes zero (thin data) or no metric improves or bucket worse.")
        print("  Candidate to re-run at week 12.")

    # Write results JSON
    result = {
        "warmup_week": warmup,
        "test_weeks": test_weeks,
        "window": WINDOW,
        "cap_margin": CAP_MARGIN,
        "cap_total": CAP_TOTAL,
        "per_week": [
            {"week": pw[0], "n": pw[1], "margin_mae_raw": float(pw[2]), "margin_mae_corr": float(pw[3]),
             "total_mae_raw": float(pw[4]), "total_mae_corr": float(pw[5]),
             "margin_correction": float(pw[6]), "total_correction": float(pw[7])}
            for pw in per_week
        ],
        "pooled": {
            "n": len(all_rows),
            "margin_mae_raw": float(raw_m.mean()), "margin_mae_corr": float(corr_m.mean()),
            "margin_delta_ci": [float(ci_m[0]), float(ci_m[1])],
            "total_mae_raw": float(raw_t.mean()), "total_mae_corr": float(corr_t.mean()),
            "total_delta_ci": [float(ci_t[0]), float(ci_t[1])],
        },
        "ship": bool(ship),
    }
    with open("docs/cfb_league_residual_ship_test.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults written to docs/cfb_league_residual_ship_test.json")

    return 0 if True else 1  # exit 0 regardless; ship decision is in the doc


if __name__ == "__main__":
    sys.exit(main())
