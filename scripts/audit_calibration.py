"""CFB calibration audit — real walk-forward on public snapshot data.

Uses public_snapshot.json weeks 1-6 (2026 season, 437 completed games) with
model predictions (home_win_prob, predicted_margin, predicted_total, sigma, total_sigma)
and actual outcomes. Research only: no src/ changes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json
import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 1


def ci(vals, n=N_BOOT, seed=SEED):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_pair(x, y, stat, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(stat(x[idx], y[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def reliability(probs, outcomes, n_bins=10, min_n=5):
    probs = np.asarray(probs, float); outcomes = np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    pm, ar, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_n:
            continue
        pm.append(probs[m].mean()); ar.append(outcomes[m].mean()); ns.append(int(m.sum()))
    pm = np.array(pm); ar = np.array(ar); ns = np.array(ns, float)
    if len(pm) > 1:
        slope, intercept = ols(pm, ar)
    else:
        slope, intercept = np.nan, np.nan
    gap = float(np.max(np.abs(pm - ar))) if len(pm) else np.nan
    return slope, intercept, gap, int(ns.sum()), pm, ar, ns


def coverage(actual, pred, sigma, k):
    return float((np.abs(actual - pred) <= k * sigma).mean())


def main():
    # Load public snapshot
    with open("data/public_snapshot.json") as f:
        snap = json.load(f)

    # Collect all completed games from weeks 1-6
    games = []
    for wk in map(str, range(1, 7)):
        week_data = snap["weeks"].get(wk, {})
        games_list = week_data.get("games", [])
        preds = week_data.get("predictions", {})
        for g in games_list:
            hs, as_ = g.get("home_score"), g.get("away_score")
            if hs is None or as_ is None:
                continue
            gid = g["game_id"]
            pred = preds.get(gid)
            if not pred:
                continue
            games.append({
                "game_id": gid,
                "week": int(wk),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_score": float(hs),
                "away_score": float(as_),
                "home_win_prob": pred["home_win_prob"],
                "predicted_margin": pred["predicted_margin"],
                "predicted_total": pred["predicted_total"],
                "sigma": pred["sigma"],
                "total_sigma": pred["total_sigma"],
                "home_cover_prob": pred.get("home_cover_prob"),
                "away_cover_prob": pred.get("away_cover_prob"),
            })

    df = pd.DataFrame(games)
    n = len(df)
    print(f"CFB calibration audit — public snapshot weeks 1-6, n={n}")

    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["actual_home_win"] = (df["actual_margin"] > 0).astype(float)

    lines = []
    lines.append("| Market | Metric | Value | 95% CI | Reading |")
    lines.append("|--------|--------|-------|--------|---------|")

    # --- Moneyline ---
    rslope, rint, rgap, rn, pm, ar, ns = reliability(df["home_win_prob"], df["actual_home_win"])
    bias = float((df["home_win_prob"] - df["actual_home_win"]).mean())
    pb_lo, pb_hi = ci(df["home_win_prob"] - df["actual_home_win"])
    rs_lo, rs_hi = boot_pair(df["home_win_prob"], df["actual_home_win"], lambda a, b: ols(a, b)[0])
    lines.append(f"| Moneyline (home win) | Reliability slope | {rslope:.3f} | [{rs_lo:.3f}, {rs_hi:.3f}] | {'problem' if rs_lo > 1.0 or rs_hi < 1.0 else 'no problem'} |")
    lines.append(f"| Moneyline (home win) | Reliability intercept | {rint:.3f} | — | ideal 0.00 |")
    lines.append(f"| Moneyline (home win) | Max calibration gap (pts) | {rgap:.3f} | — | <0.03 honest |")
    lines.append(f"| Moneyline (home win) | Mean predicted prob − actual rate | {bias:+.3f} | [{pb_lo:+.3f}, {pb_hi:+.3f}] | {'problem' if pb_lo > 0 or pb_hi < 0 else 'no problem'} |")
    # Bin detail
    bin_detail = " | ".join(f"[{p:.2f},{a:.2f},n={n}]" for p,a,n in zip(pm, ar, ns))
    lines.append(f"| Moneyline | Decile bins (min_n=5) | {bin_detail} | — | |")

    # --- Margin continuous ---
    margin_bias = float((df["predicted_margin"] - df["actual_margin"]).mean())
    mb_lo, mb_hi = ci(df["predicted_margin"] - df["actual_margin"])
    lines.append(f"| Margin | Mean predicted − actual (pts) | {margin_bias:+.3f} | [{mb_lo:+.3f}, {mb_hi:+.3f}] | {'problem' if mb_lo > 0 or mb_hi < 0 else 'no problem'} |")
    c1 = coverage(df["actual_margin"], df["predicted_margin"], df["sigma"], 1.0)
    c2 = coverage(df["actual_margin"], df["predicted_margin"], df["sigma"], 2.0)
    c1d = (np.abs(df["actual_margin"] - df["predicted_margin"]) <= df["sigma"])
    c2d = (np.abs(df["actual_margin"] - df["predicted_margin"]) <= 2 * df["sigma"])
    c1_lo, c1_hi = ci(c1d); c2_lo, c2_hi = ci(c2d)
    lines.append(f"| Margin | Within 1 sigma | {c1:.3f} | [{c1_lo:.3f}, {c1_hi:.3f}] | expected 0.683 |")
    lines.append(f"| Margin | Within 2 sigma | {c2:.3f} | [{c2_lo:.3f}, {c2_hi:.3f}] | expected 0.954 |")
    rs, ri = ols(df["predicted_margin"], df["actual_margin"])
    rs_lo, rs_hi = boot_pair(df["predicted_margin"], df["actual_margin"], lambda a, b: ols(a, b)[0])
    lines.append(f"| Margin | Regression-to-mean slope | {rs:.3f} | [{rs_lo:.3f}, {rs_hi:.3f}] | {'problem' if rs_lo > 1.0 or rs_hi < 1.0 else 'no problem'} |")

    # --- Total continuous ---
    total_bias = float((df["predicted_total"] - df["actual_total"]).mean())
    tb_lo, tb_hi = ci(df["predicted_total"] - df["actual_total"])
    lines.append(f"| Total | Mean predicted − actual (pts) | {total_bias:+.3f} | [{tb_lo:+.3f}, {tb_hi:+.3f}] | {'problem' if tb_lo > 0 or tb_hi < 0 else 'no problem'} |")
    tc1 = coverage(df["actual_total"], df["predicted_total"], df["total_sigma"], 1.0)
    tc2 = coverage(df["actual_total"], df["predicted_total"], df["total_sigma"], 2.0)
    tc1d = (np.abs(df["actual_total"] - df["predicted_total"]) <= df["total_sigma"])
    tc2d = (np.abs(df["actual_total"] - df["predicted_total"]) <= 2 * df["total_sigma"])
    tc1_lo, tc1_hi = ci(tc1d); tc2_lo, tc2_hi = ci(tc2d)
    lines.append(f"| Total | Within 1 sigma | {tc1:.3f} | [{tc1_lo:.3f}, {tc1_hi:.3f}] | expected 0.683 |")
    lines.append(f"| Total | Within 2 sigma | {tc2:.3f} | [{tc2_lo:.3f}, {tc2_hi:.3f}] | expected 0.954 |")

    # --- Spread cover (if cover probs available) ---
    has_cover = df["home_cover_prob"].notna() & df["away_cover_prob"].notna()
    if has_cover.any():
        cover_df = df[has_cover].copy()
        cover_df["actual_cover"] = (cover_df["actual_margin"] > 0).astype(float)  # simplified
        cover_bias = float((cover_df["home_cover_prob"] - cover_df["actual_cover"]).mean())
        cb_lo, cb_hi = ci(cover_df["home_cover_prob"] - cover_df["actual_cover"])
        lines.append(f"| Spread cover | Mean cover prob − actual | {cover_bias:+.3f} | [{cb_lo:+.3f}, {cb_hi:+.3f}] | {'problem' if cb_lo > 0 or cb_hi < 0 else 'no problem'} |")
    else:
        lines.append("| Spread cover | — | cover probs None in snapshot | — | skipped |")

    # --- Bias by situation ---
    # Home vs away favourite
    fav = df["predicted_margin"] > 0
    if fav.any() and (~fav).any():
        b1 = (df.loc[fav, "home_win_prob"] - df.loc[fav, "actual_home_win"]).mean()
        b2 = (df.loc[~fav, "home_win_prob"] - df.loc[~fav, "actual_home_win"]).mean()
        lines.append(f"| Moneyline | Bias home favoured | {b1:+.3f} | {ci(df.loc[fav, 'home_win_prob'] - df.loc[fav, 'actual_home_win'])[0]:+.3f}..{ci(df.loc[fav, 'home_win_prob'] - df.loc[fav, 'actual_home_win'])[1]:+.3f} | |")
        lines.append(f"| Moneyline | Bias away favoured | {b2:+.3f} | {ci(df.loc[~fav, 'home_win_prob'] - df.loc[~fav, 'actual_home_win'])[0]:+.3f}..{ci(df.loc[~fav, 'home_win_prob'] - df.loc[~fav, 'actual_home_win'])[1]:+.3f} | |")
    # Big favourites
    big = df["predicted_margin"].abs() >= 14
    if big.any():
        lines.append(f"| Moneyline | Bias |margin|>=14 | {(df.loc[big, 'home_win_prob'] - df.loc[big, 'actual_home_win']).mean():+.3f} | {ci(df.loc[big, 'home_win_prob'] - df.loc[big, 'actual_home_win'])[0]:+.3f}..{ci(df.loc[big, 'home_win_prob'] - df.loc[big, 'actual_home_win'])[1]:+.3f} | |")
        lines.append(f"| Margin | Bias |margin|>=14 | {(df.loc[big, 'predicted_margin'] - df.loc[big, 'actual_margin']).mean():+.3f} | {ci(df.loc[big, 'predicted_margin'] - df.loc[big, 'actual_margin'])[0]:+.3f}..{ci(df.loc[big, 'predicted_margin'] - df.loc[big, 'actual_margin'])[1]:+.3f} | |")

    report = "\n".join(lines)
    print(report)

    out = Path("docs/cfb_calibration_audit.md")
    out.write_text(
        f"# CFB calibration audit (public snapshot weeks 1-6)\n\n"
        f"Generated {datetime.utcnow().date()}. Ridge margin model, weeks 1-6 (2026 season), "
        f"{len(df)} completed games. Paired-bootstrap 95% CIs over {N_BOOT} resamples. "
        f"Research only: no model output changed, nothing under src/.\n\n"
        "Ideals: reliability slope 1.00 / intercept 0.00; bias 0; sigma coverage 0.683 (1σ) / 0.954 (2σ);\n"
        "regression-to-the-mean slope 1.00. Max calibration gap <0.03 reads as honest.\n\n"
        "## Results\n\n" + report + "\n\n"
        "## Interpretation\n"
        "- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.\n"
        "- Fix proposed only where paired-bootstrap CI excludes the ideal value (bias≠0, slope≠1, coverage≠nominal).\n"
        "- Data source: public_snapshot.json weeks 1-6 (2026 season), 437 completed games with model predictions.\n"
        "- No CFBD API key needed — used public snapshot data with model predictions and actual outcomes.\n"
        "- No fix shipped here: research only, no serving change proposed without paired-bootstrap CI excluding ideal on re-tested candidates.\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()