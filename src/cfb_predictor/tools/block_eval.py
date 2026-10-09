"""Decide whether a CFB feature block earns its place. Paired bootstrap on identical held-out games."""
from __future__ import annotations

import argparse

import numpy as np

from cfb_predictor.data import cfbd_advanced
from cfb_predictor.data import games
from cfb_predictor.evaluate import walk_forward as wf
from cfb_predictor.features import build as feature_build


def paired_bootstrap(a, b, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of mean(a - b). Positive means b is better when lower is better."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibration_gap(probs, y, buckets: int = 5) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y, float)
    edges = np.quantile(probs, np.linspace(0, 1, buckets + 1))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, buckets - 1)
    gaps = [abs(probs[idx == b].mean() - y[idx == b].mean()) for b in range(buckets) if (idx == b).any()]
    return float(max(gaps)) if gaps else 0.0


def _per_game(folds, candidate):
    errs, briers, probs, ys = [], [], [], []
    for fold in folds:
        preds, sigma = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        margin = fold["val_df"]["margin"].to_numpy()
        p = np.array([wf.game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        errs.append(np.abs(preds - margin))
        y = (margin > 0).astype(float)
        briers.append((p - y) ** 2)
        probs.append(p)
        ys.append(y)
    return tuple(np.concatenate(x) for x in (errs, briers, probs, ys))


def _per_fold_gap(folds, candidate) -> list[float]:
    """Per-fold calibration gaps so the gap difference can be paired and bootstrapped."""
    gaps = []
    for fold in folds:
        preds, sigma = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        margin = fold["val_df"]["margin"].to_numpy()
        p = np.array([wf.game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        gaps.append(calibration_gap(p, (margin > 0).astype(float)))
    return gaps


def paired_bootstrap_gap(base_gaps, block_gaps, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(base_gaps, float) - np.asarray(block_gaps, float)
    if d.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_recent_cfb_data(start_year: int = 2014, end_year: int = None, from_disk: bool = True):
    """Games plus the `epa` block's efficiency aux, read from the committed JSON.

    `from_disk=False` re-derives the frame from the raw pulled rows; either way no API
    call is spent, which keeps the block_eval run reproducible and budget-free.
    """
    from datetime import datetime

    end_year = end_year if end_year is not None else datetime.now().year
    seasons = list(range(start_year, end_year + 1))

    print(f"Loading CFB data for seasons {seasons}...")
    games_df = games.load_training_data(seasons)
    efficiency = cfbd_advanced.efficiency_frame_from_disk(years=seasons)
    if efficiency.empty:
        raise ValueError(
            f"no CFBD advanced stats on disk for {seasons}; run "
            f"python -m cfb_predictor.data.cfbd_advanced --years {seasons[0]} {seasons[-1]}"
        )
    print(f"efficiency rows: {len(efficiency)} covering seasons {sorted(efficiency['season'].dropna().astype(int).unique())}")
    # Only keep games whose season has efficiency coverage, so base and block see the same games.
    covered = set(efficiency["season"].dropna().astype(int))
    dropped = sorted(set(seasons) - covered)
    if dropped:
        print(f"Warning: no efficiency for seasons {dropped}; dropping those games from both runs")
        games_df = games_df[games_df["season"].isin(covered)].reset_index(drop=True)

    aux = feature_build.Aux(efficiency=efficiency if not efficiency.empty else None)
    return games_df, aux


def _fit_and_probs(folds, candidate, target: str = "margin"):
    """(per-game abs error, brier, probs, labels) pooled across folds for one target."""
    errs, briers, probs, ys = [], [], [], []
    for fold in folds:
        preds = wf._fit_predict(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"], target)
        actual = fold["val_df"][target].to_numpy(float)
        errs.append(np.abs(preds - actual))
        if target == "margin":
            sigma = wf._honest_sigma(candidate, fold["train_df"], fold["feature_cols"], target)
            from cfb_predictor.models import game_outcome
            p = np.array([game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
            y = (actual > 0).astype(float)
            briers.append((p - y) ** 2)
            probs.append(p)
            ys.append(y)
    return (np.concatenate(errs) if errs else np.array([]),
            np.concatenate(briers) if briers else np.array([]),
            np.concatenate(probs) if probs else np.array([]),
            np.concatenate(ys) if ys else np.array([]))


def evaluate_block(games_df, aux, block: str, candidate: str = "ridge"):
    """Returns (result, None) or (None, reason). A block the pipeline cannot build is
    reported, never scored as a no-op -- that is how 'not wired' stays visible."""
    blocks = (block,)
    if block in ("epa", "priors") and (aux is None or getattr(aux, "efficiency", None) is None):
        return None, f"aux.efficiency is required for block {block!r}"
    try:
        base_folds = wf.prepare_folds(games_df)
        withb_folds = wf.prepare_folds(games_df, blocks=blocks, aux=aux)
    except ValueError as e:
        return None, f"block {block!r} cannot be built: {e}"
    base = _per_game(base_folds, candidate)
    withb = _per_game(withb_folds, candidate)
    if len(base[0]) != len(withb[0]):
        return None, f"block {block} drops games: {len(base[0])} vs {len(withb[0])}"
    mae_ci = paired_bootstrap(base[0], withb[0])
    brier_ci = paired_bootstrap(base[1], withb[1])
    gap_base = calibration_gap(base[2], base[3])
    gap_block = calibration_gap(withb[2], withb[3])
    gap_ci = paired_bootstrap_gap(_per_fold_gap(base_folds, candidate),
                                  _per_fold_gap(withb_folds, candidate))
    return {
        "block": block, "n_games": int(len(base[0])),
        "mae_delta": float(base[0].mean() - withb[0].mean()), "mae_ci": mae_ci,
        "brier_delta": float(base[1].mean() - withb[1].mean()), "brier_ci": brier_ci,
        "gap_base": gap_base, "gap_block": gap_block, "gap_ci": gap_ci,
        "clears": bool(mae_ci[0] > 0 and brier_ci[0] > 0 and gap_ci[0] > 0),
    }, None


def _fmt(ci):
    lo, hi = ci
    return f"[{lo:+.5f}, {hi:+.5f}]" if all(-1 < v < 1 for v in ci) else f"[{lo:.4f}, {hi:.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CFB feature blocks")
    parser.add_argument("--blocks", nargs="+", default=["epa"])
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--candidate", type=str, default="ridge")
    args = parser.parse_args()

    games_df, aux = load_recent_cfb_data(args.start_year, args.end_year)
    if games_df.empty:
        print("Error: no games loaded")
        return 1
    print(f"Loaded {len(games_df)} games. Evaluating: {', '.join(args.blocks)}\n")

    results = {}
    for block in args.blocks:
        print(f"Evaluating block: {block}...")
        try:
            res, err = evaluate_block(games_df, aux, block, args.candidate)
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        if res is None:
            print(f"  -> NOT EVALUATED: {err}")
            results[block] = None
        else:
            results[block] = res

    print("\n" + "=" * 84)
    print("CFB BLOCK EVALUATION RESULTS")
    print("=" * 84)
    hdr = f"{'Block':<10} {'N':>6} {'MAE d':>9} {'MAE 95% CI':>21} {'Brier d':>9} {'Brier 95% CI':>21} {'Gap d CI':>21} {'Clears':>6}"
    print(hdr)
    print("-" * len(hdr))
    for block, r in results.items():
        if r is None:
            continue
        print(f"{r['block']:<10} {r['n_games']:>6} {r['mae_delta']:>+9.5f} {_fmt(r['mae_ci']):>21} "
              f"{r['brier_delta']:>+9.5f} {_fmt(r['brier_ci']):>21} {_fmt(r['gap_ci']):>21} {'Yes' if r['clears'] else 'No':>6}")
    for block in args.blocks:
        if results.get(block) is None:
            print(f"{block:<10}  not evaluated")
    print("=" * 84)
    print("Sign convention: delta = base - block, so POSITIVE means the block is better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
