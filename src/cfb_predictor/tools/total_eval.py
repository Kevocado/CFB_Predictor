"""Does a ridge + opponent-adjusted-EPA TOTAL model beat the XGBoost total model? Repo walk-forward, paired bootstrap.

Same folds and same held-out games for both. Promotion bar: the 95% interval of the MAE improvement excludes 0 AND the
total-bias gap (largest |mean error| across prediction quintiles) is not widened (interval of the change < 0 or the
point change <= 0 with the interval not above 0).
"""
from __future__ import annotations

import numpy as np

from cfb_predictor.evaluate import walk_forward as wf
from cfb_predictor.features import build as feature_build
from cfb_predictor.tools import block_eval as be


def bias_gap(pred, actual, buckets: int = 5) -> float:
    q = np.quantile(pred, np.linspace(0, 1, buckets + 1))
    i = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, buckets - 1)
    return max(abs((pred[i == k] - actual[i == k]).mean()) for k in range(buckets) if (i == k).any())


def paired_bias_gap(pa, pb, y, n: int = 300, seed: int = 1) -> tuple[float, float]:
    """95% interval of gap(b) - gap(a); negative means b is less biased."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(bias_gap(pb[i], y[i]) - bias_gap(pa[i], y[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _pooled(folds, candidate):
    pred = np.concatenate([wf._fit_predict(candidate, f["train_df"], f["val_df"], f["feature_cols"], "total_points") for f in folds])
    act = np.concatenate([f["val_df"]["total_points"].to_numpy(float) for f in folds])
    return pred, act


def evaluate(games_df, aux, block: str = "adj_epa", candidate: str = "ridge") -> dict:
    base_p, y = _pooled(wf.prepare_folds(games_df), "xgb")
    new_p, y2 = _pooled(wf.prepare_folds(games_df, blocks=(block,), aux=aux), candidate)
    assert np.array_equal(y, y2), "folds differ: the two models must see identical held-out games"
    eb, en = np.abs(base_p - y), np.abs(new_p - y)
    mae_ci = be.paired_bootstrap(eb, en)
    gap_ci = paired_bias_gap(base_p, new_p, y)
    gb, gn = bias_gap(base_p, y), bias_gap(new_p, y)
    return {
        "n": len(y), "mae_base": float(eb.mean()), "mae_new": float(en.mean()), "mae_gain": float((eb - en).mean()),
        "mae_ci": mae_ci, "gap_base": float(gb), "gap_new": float(gn), "gap_change_ci": gap_ci,
        "clears": bool(mae_ci[0] > 0 and gn <= gb),
    }


def main() -> int:
    games_df, aux = be.load_recent_cfb_data(2014, 2025)
    games_df["game_id"] = games_df["game_id"].astype(str)
    r = evaluate(games_df, aux)
    print(f"N={r['n']}  total MAE xgb {r['mae_base']:.4f} -> ridge+adj_epa {r['mae_new']:.4f}  "
          f"gain {r['mae_gain']:+.4f} CI [{r['mae_ci'][0]:+.4f}, {r['mae_ci'][1]:+.4f}]")
    print(f"total-bias gap {r['gap_base']:.3f} -> {r['gap_new']:.3f}  change CI [{r['gap_change_ci'][0]:+.3f}, {r['gap_change_ci'][1]:+.3f}]")
    print("CLEARS" if r["clears"] else "DOES NOT SHIP")
    return 0 if r["clears"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
