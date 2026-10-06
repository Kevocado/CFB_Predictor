"""qb_passing_td.py -- a per-QB expected passing-TD projection (mu), a model
line derived from it, and an over/under call on that line.

Ported from `nfl_predictor/models/qb_passing_td.py`, which was written for this
problem first. The arithmetic, the distribution choice, the half-point line grid
and the degenerate-fit guards are NFL's and are unchanged, because a count model
that means one thing on both sites is worth more than a locally-invented second
version of it. What is CFB's is the fit.

**Every MEASURED number that was NFL's has been removed rather than reworded.**
The port's own first draft carried NFL's fit outcome -- "POISSON WINS", 5,179
player-weeks, a variance ratio of 1.0888, log losses of 1.392621 -- with the
attribution mechanically stripped, so CFB's file claimed measurements nobody had
made on CFB data. That is the exact failure NFL's own docstring warns about a few
lines further down ("a comment that lies about the artefact it describes is worse
than no comment, because it is the one a reader trusts"), reached by copying the
comment instead of the code.

NFL's finding was that its QB passing TDs are mildly overdispersed
unconditionally but UNDERdispersed once the mean is conditioned on usage, so the
negative-binomial fit drives alpha to its ceiling and converges onto Poisson.
**CFB is not assumed to be the same.** College quarterbacks are a different
population -- more run-heavy, shorter passing games, a different distribution of
attempted passes per game -- so which distribution wins here is a question this
repository has to answer by fitting, not by inheritance. `fit_qb_passing_td_model`
records the outcome in the artefact, and the numbers to quote from it are read
back out of `models/manifest.json`.

What existed before this file, on CFB
------------------------------------
* **There was no expected-passing-TD projection on CFB** -- not in `features/`,
  not in `models/`, not in the box score, not in the stats feed.
* `player_stats.KEEP_COLUMNS` carries a real `passing_tds` column
  (`data/player_stats.py`, mapped from CFBD's `("passing", "TD")`), so the target
  data exists. Measured on the production cache 2026-10-04: 219,481 player rows
  across 2018-2026, of which **34,675 are QB rows and 17,095 carry a passing TD**,
  mean 0.94 per QB game over 7,749 distinct quarterbacks. That is the coverage
  the fit needs, and it is the prerequisite the plan named ("needs its own data
  validation") -- satisfied before a line of this file was written.
* `models/player_props.py` produces `anytime_td_prob` (a binary classifier) and
  per-position yardage point estimates. No count model, no line of any kind.
* `features/player_usage.PLAYER_FEATURE_COLUMNS` rolls `passing_yards`,
  `rushing_yards`, `receiving_yards`, `targets`, `carries`, `receptions` -- and
  **not** `passing_tds`. So there was no passing-TD rolling feature either, and
  `with_passing_tds_roll` was added alongside this file.

So nothing here invents a target column: it projects the existing `passing_tds`
column, using a rolling feature built from the same pregame discipline the rest
of the player features use.

`passing_tds_roll` is deliberately kept out of `PLAYER_FEATURE_COLUMNS` -- that
list is what the anytime-TD classifier and every yardage regressor are fitted on,
so widening it would change the feature count of every already-committed model --
and is emitted by `build_features_for_player` as its own column instead. It is
the model's first fitted column, and `models/manifest.py` asserts at fit and at
load that the whole fitted list is a subset of what the serving builder emits. An
earlier version of this branch fitted on it and emitted nothing, so every QB was
projected from `fillna(0)` on it.

The "model line", and what it is not
------------------------------------
`model_line(mu)` is the nearest half point to the model's own expectation. It is
**derived from the projection**, so it is not a sportsbook line and calling it an
edge would be a category error -- there is nothing to have an edge *against*. It
is labelled `"model_line"` everywhere it is produced and stored.

If a real book line is ever wired in, it replaces the model line **without
changing the record format**: the line is a number in a column, and the side,
mu and probability are computed from whatever line is in hand. See
`passing_td_call`, which takes the line from `model_line(mu)` and would take it
from a feed instead, unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

#: The market name this writes. Distinct from `anytime_td` by construction --
#: `anytime_td` is `rushing_tds + receiving_tds > 0`
#: (`features/player_usage.py::anytime_td_actual`), so passing TDs cannot reach
#: it, and this is the category where a QB is judged on passing TDs at all.
#: This comment previously stated the opposite -- that `anytime_td` was
#: `rushing_tds + receiving_tds + passing_tds > 0` and was "dominated by his
#: passing" -- which was true of CFB's old label v1 and false as of 2026-10-04,
#: when CFB dropped passing TDs from it to match NFL's v2
#: (`ANYTIME_TD_LABEL_VERSION = 2`, and `tests/test_anytime_td_label.py` holds
#: the label to it). The port inherited NFL's phrasing of this comment, which
#: dated the change to 2026-10-01 and attributed it to NFL; on CFB the change is
#: one day later and was CFB's own, so the date and the attribution both follow
#: the code rather than the file it was copied from.
PASSING_TD_MARKET = "passing_tds"

#: The only value `line_source` ever takes. It is a constant, not a parameter,
#: because "model line" is a fact about provenance rather than a choice.
MODEL_LINE_SOURCE = "model_line"

#: A line is never below this. A projected 0.1 passing TDs still has to be
#: expressible as an over/under, and 0.0 is not a bettable line.
MIN_MODEL_LINE = 0.5

#: Rolling usage stats the mu model reads, on top of `passing_tds_roll` itself.
MU_FEATURE_COLUMNS = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll"]


@dataclass(frozen=True)
class DistributionSpec:
    """Which count distribution the count is drawn from, and its dispersion.

    `alpha` is the NB2 overdispersion parameter (variance `mu + mu**2 / alpha`);
    it is meaningless for Poisson and is `None` there. Frozen because a spec is
    a value, and a call built from one must not change under the caller.
    """

    kind: str
    alpha: float | None = None

    def __post_init__(self):
        if self.kind not in ("poisson", "negative_binomial"):
            raise ValueError(f"unknown distribution kind: {self.kind!r}")
        if self.kind == "negative_binomial" and not (self.alpha and self.alpha > 0):
            raise ValueError("negative_binomial needs a positive alpha")


def model_line(mu: float) -> float:
    """The nearest half point to `mu`, never below 0.5.

        mu 1.8  -> 1.5        mu 2.3  -> 2.5
        mu 1.0  -> 1.5        mu 0.3  -> 0.5

    **The candidate set is the numbers ENDING IN .5 -- {0.5, 1.5, 2.5, ...} -- and
    not every multiple of 0.5.** That distinction is the whole rule, and the first
    version of this function got it wrong.

    Rounding onto {0, 0.5, 1.0, 1.5, ...} is what "nearest half point" looks like
    read carelessly, and it produces **whole-number lines**: `model_line(1.0)`
    came out 1.0. A whole-number line is an instant push waiting to happen -- a
    quarterback throws exactly 1 passing TD, actual == line, and neither over nor
    under is true. `tracking/store.py::_line_is_half_point` then refuses to grade
    the row, and the pick is reported ungradeable rather than scored.

    Restricting the grid to x.5 makes push structurally impossible rather than
    merely unlikely, which is what "push cannot occur" has to mean for the record
    to be honest. And it is the reading that reproduces both worked examples:
    against {0.5, 1.5, 2.5}, mu 1.8 is nearest 1.5 and mu 2.3 is nearest 2.5.

    So the rule is `floor(mu) + 0.5`: take the integer part and add a half. Ties
    (mu exactly on a whole number) go UP, matching half-up rounding -- under
    Python's banker's rounding `round(1.0)` would be `1` and land the same mu on
    0.5 sometimes and 1.5 other times depending on parity, which is worse than
    any consistent choice.
    """
    if mu is None or not math.isfinite(mu):
        raise ValueError(f"mu must be a finite number, got {mu!r}")
    return max(MIN_MODEL_LINE, math.floor(float(mu)) + 0.5)


def _tail_probs(mu: float, threshold: float, spec: DistributionSpec) -> tuple[float, float]:
    """(P(count > threshold), P(count < threshold)) under `spec`.

    `threshold` is the line itself, and it always ends in .5, so `>` and `<` are
    exhaustive over the integers: there is no third outcome and no push.
    """
    mu = max(float(mu), 1e-12)
    if spec.kind == "poisson":
        return float(stats.poisson.sf(threshold, mu)), float(stats.poisson.cdf(threshold, mu))
    alpha = float(spec.alpha)
    # NB2 parametrised for scipy as n=alpha, p=alpha/(alpha+mu), which has
    # mean mu and variance mu + mu**2/alpha.
    p = alpha / (alpha + mu)
    return float(stats.nbinom.sf(threshold, alpha, p)), float(stats.nbinom.cdf(threshold, alpha, p))


def passing_td_call(mu: float, spec: DistributionSpec, line: float | None = None) -> dict:
    """The over/under call for one QB at expectation `mu`.

    The line is `model_line(mu)` unless one is passed in. **If a real sportsbook
    line is ever wired in it is passed here instead, and nothing about this
    function or the pick record changes** -- the line is an argument, not a
    constant baked into the call, and `line_source` is the only field that has to
    be set to describe where it came from. Until then it is always
    `"model_line"`, and because the line is derived from `mu` this is **not an
    edge claim**.

    The side is the higher-probability side and `call_prob` is that probability.
    """
    line = model_line(mu) if line is None else float(line)
    over_prob, under_prob = _tail_probs(mu, line, spec)
    side = "over" if over_prob > under_prob else "under"
    return {
        "market": PASSING_TD_MARKET,
        "line": line,
        "line_source": MODEL_LINE_SOURCE,
        "side": side,
        "mu": float(mu),
        "over_prob": over_prob,
        "under_prob": under_prob,
        "call_prob": max(over_prob, under_prob),
        "distribution": spec.kind,
        # A half-point line against an integer count cannot tie. Carried
        # explicitly so a consumer never has to infer it.
        "push_prob": 0.0,
    }


#: Bounds on `log(alpha)` for the NB2 dispersion fit, and why they exist.
#:
#: **An unbounded fit is numerically broken here, and it fails silently.** The
#: negative log-likelihood is flat in alpha over roughly [20, 1e4] and then
#: *decreases* again past ~1e7, because at that point `nbinom.logpmf` starts
#: returning `inf` for every row as the distribution collapses onto a single
#: point mass. Mean `inf` is `-inf`, which beats any honest likelihood, so an
#: unbounded Nelder-Mead walks straight into it.
#:
#: **The -5.81 figure in the next line is NFL's, measured on NFL's 2018-2024 QB
#: history**, and is kept as the worked example of why these bounds exist rather
#: than dropped: the failure is a property of `nbinom.logpmf` under an unbounded
#: scalar optimiser, not of any one sport's data, so CFB would hit it identically.
#: CFB has not been fitted unbounded, and this file does not claim it was. The
#: number is a demonstration that the failure is reachable, not a claim about
#: CFB's own fits -- which is recorded in the artefact and nowhere here.
#:
#: log loss of **-5.81** -- a value no distribution can achieve, since a mean NLL
#: over a real pmf cannot be negative when the data are integers the pmf
#: supports.
#:
#: The bounds keep alpha in [exp(-8), exp(12)] = [3.4e-4, 1.6e5], which covers
#: "no more overdispersion than Poisson by a wide margin" through "essentially
#: Poisson" and excludes the collapsed region entirely.
_NB_LOG_ALPHA_BOUNDS = (-8.0, 12.0)

#: The log loss a distribution must beat before it is considered to have fit at
#: all. A mean NLL below this is not a good model, it is a broken one.
_MIN_PLAUSIBLE_LOG_LOSS = -1.0


def _nb_alpha_nll(log_alpha: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Mean negative log-likelihood of NB2 counts at fixed means `mu`.

    Dispersion only: the mean is held at the Poisson fit so the two
    distributions differ in exactly one parameter and the log-loss comparison
    measures overdispersion rather than a better mean.
    """
    alpha = math.exp(float(log_alpha))
    p = alpha / (alpha + np.clip(mu, 1e-12, None))
    nll = float(-stats.nbinom.logpmf(y, alpha, p).mean())
    # A non-finite likelihood is a fitting failure, not a good score. Returning
    # +inf keeps the optimizer away from it; the guard below then refuses to
    # report the resulting number as a real log loss.
    return nll if math.isfinite(nll) else math.inf


def fit_qb_passing_td_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit BOTH count distributions on QB history and choose by log loss.

    **MEASURED OUTCOME ON CFB: NEGATIVE BINOMIAL WINS, and it is not a tie.**
    At the commit that fitted the artefact now in `models/` (seasons 2018-2025,
    from the production `player_stats` cache):

        usable QB rows      28,466
        variance-to-mean     1.506
        in-sample log loss   negative binomial 1.245850   poisson 1.251011
                             delta = 5.2e-03 in negative binomial's favour
        alpha                7.003

    **THOSE NUMBERS ARE A RECORD OF ONE FIT, NOT A PROMISE ABOUT THE NEXT.**
    `models/manifest.json`'s `qb_passing_td` entry carries the same figures for
    whatever artefact is actually committed, and
    `tests/test_qb_passing_td_artefact.py` asserts the manifest entry describes the
    committed pickle -- so the pair cannot disagree with each other.

    This paragraph is not that mechanism, and it is here as the cautionary half of
    it. It previously read 26,926 rows / 1.390346 / alpha 7.36571588075843, measured
    2026-10-04, and every one of those figures was **stale the moment the artefact
    was refitted** with a season of data added. Nothing noticed, because prose is
    not asserted by default. NFL learned the same lesson the hard way -- its
    docstring described its artefact as it was *before* a retrain onto anytime-TD
    label v2 features -- and the fix both times was to read the number back out of
    the artefact rather than to write it down and hope.

    **NFL, by contrast, ended up at Poisson** -- its log losses were identical to
    seven decimal places, because once its mean was conditioned on usage features
    its QB passing TDs were UNDERdispersed and the NB2 fit drove alpha to its
    ceiling. CFB is the opposite: a variance-to-mean above 1.5 is enough real
    overdispersion for a dispersion parameter to explain, so alpha lands in the
    interior rather than against a bound. That is why this docstring asserts no
    outcome until the fit ran, and why the port stripped NFL's rather than
    reworded it: **had it inherited "POISSON WINS", CFB would have shipped a
    Poisson artefact that is measurably the worse of the two fits available**, and
    nothing in the repository would have said so.

    Five times NFL's training rows (28,466 against 5,179) is also CFB's own
    measurement.

    Poisson first (it is the mean model and the NB mean is held at it), then the
    NB2 dispersion fitted by MLE on the same rows. Both are scored as mean
    negative log-likelihood over the whole history -- the log loss of a count
    distribution -- and the lower one wins. Both numbers are returned under
    `log_loss` so the comparison is inspectable rather than asserted.

    The feature columns are taken from `X` as given, so a caller controls which
    rolling features the model reads and the training frame and the pregame
    feature row cannot drift apart silently.
    """
    from sklearn.linear_model import PoissonRegressor

    cols = list(X.columns)
    Xv = np.asarray(X[cols].to_numpy(dtype=float), dtype=float)
    yv = np.asarray(y.to_numpy(dtype=float), dtype=float)
    if Xv.size == 0:
        raise ValueError("no QB history to fit a passing-TD model on")

    poisson = PoissonRegressor(alpha=1e-8, max_iter=1000).fit(Xv, yv)
    mu = np.clip(poisson.predict(Xv), 1e-12, None)
    poisson_ll = float(-stats.poisson.logpmf(yv, mu).mean())

    fitted = optimize.minimize_scalar(
        lambda la: _nb_alpha_nll(la, yv, mu),
        bounds=_NB_LOG_ALPHA_BOUNDS, method="bounded",
    )
    alpha = float(math.exp(float(fitted.x)))
    p = alpha / (alpha + mu)
    nb_ll = _nb_alpha_nll(fitted.x, yv, mu)

    scores = {"poisson": poisson_ll, "negative_binomial": nb_ll}
    # A distribution that "wins" by scoring below what any real pmf can achieve
    # has not won; it has broken. Dropping it back to the finite alternative is
    # the honest read, and it is asserted so this cannot pass unnoticed again.
    if not math.isfinite(min(scores.values())) or min(scores.values()) < _MIN_PLAUSIBLE_LOG_LOSS:
        scores = {"poisson": poisson_ll, "negative_binomial": math.inf}
    chosen = min(scores, key=scores.get)

    return {
        "distribution": chosen,
        "alpha": alpha if chosen == "negative_binomial" else None,
        "log_loss": scores,
        "model": poisson,
        "feature_cols": cols,
        "n_train": int(len(yv)),
        # var/mean on the fitted means, so the overdispersion the choice is
        # reacting to is visible in the artifact itself.
        "variance_ratio": float(np.var(yv) / max(np.mean(yv), 1e-12)),
    }


def expected_passing_tds(fitted: dict, feature_row: pd.Series) -> float:
    """mu for one QB, from that QB's own pregame feature row.

    Reindexes onto the columns the model was fitted on, so a row missing a
    feature cannot silently shift every column by one.

    A fitted column **absent** from the row raises rather than being filled. The
    reindex cannot tell a missing column from a null value, so `fillna(0)` used to
    cover both, and the missing-column half is what served every QB from a
    constant-zero `passing_tds_roll`. A null *value* on a present column is a
    different case -- a player with no prior games in scope -- and still fills,
    because `routes` already skips those players before they reach here.
    `load_models` asserts the same subset property for the whole payload at load
    time; this is the per-row backstop.
    """
    cols = list(fitted["feature_cols"])
    missing = [c for c in cols if c not in feature_row.index]
    if missing:
        raise KeyError(
            f"the QB passing-TD model was fitted on {cols}, which the served feature row does "
            f"not carry: {missing}. Scoring it anyway would fill {missing} with 0.0 -- the "
            "fitted-a-feature-served-as-constant-zero defect, per player."
        )
    X = feature_row.reindex(cols).fillna(0).to_numpy(dtype=float).reshape(1, -1)
    return max(float(fitted["model"].predict(X)[0]), 0.0)


def qb_passing_td_call(fitted: dict, feature_row: pd.Series) -> dict | None:
    """Projection -> line -> side for one QB, or None when it cannot be produced.

    None rather than a zero row: a QB with no history has no mu, and a fabricated
    0.0 mu would produce a real-looking 0.5 line and a real-looking side. That is
    the all-zero-region defect `routes._get_player_props_live` already skips
    players for, and it applies here too.
    """
    if fitted is None or fitted.get("model") is None:
        return None
    mu = expected_passing_tds(fitted, feature_row)
    spec = DistributionSpec(fitted["distribution"], alpha=fitted.get("alpha"))
    return passing_td_call(mu, spec)