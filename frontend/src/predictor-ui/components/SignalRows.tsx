// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * `Signal[]` as the facts block renders it: one compact row per finding, a
 * headline of at most 12 words, and a collapsed evidence line a keyboard can
 * open.
 *
 * Spec `2026-10-01-fixture-signals-design.md` §3 (the contract), §6 (the look).
 * The type below is §3 **verbatim in field and meaning**; everything after it is
 * this component's own account of what the fields may not be used to do.
 *
 * ## The one rule everything else serves
 *
 * **A figure may only reach the page if the same figure is in the row's own
 * words.** A signal is code-computed (spec §2: "Code computes the insight; the
 * AI only chooses and words it"), so the temptation is to let the visual carry a
 * number the headline does not mention — a bar 74% wide under words that never
 * said 74%. That is the "the model supplies words and ordering, never a number
 * the panel renders" rule from explainer-v2, one level up, and it is why
 * `headline` is **two** fields and not one:
 *
 *   - `headline.text` — the one line a reader reads, at most 12 words.
 *   - `headline.figures` — the numbers the visual draws.
 *
 * They are separate so the two can disagree, and `signalIsDrawn` is where that
 * disagreement is caught: **every render path checks that the figure the visual
 * draws appears in the headline's words, and refuses the row when it does not**
 * (`HeadlineFigureMismatchError` — see `assertFigureIsStated`).
 *
 * "Appears in the words" is decided by comparing NUMBERS, not by comparing words
 * or strings — `headlineStatesFigure`, and the long history of why that is not a
 * detail is on `STATED_FIGURE`. The three ways a figure can be stated wrongly
 * that a string comparison gets wrong, each found by review against this code:
 * `1%` is not `61%`; `<1%` is not `1%`; and `−1.6` is not `+1.6`.
 *
 * Merging the two into one formatted string was considered and rejected: a
 * component that *parsed* its own headline to find the figure would make the
 * assertion vacuous — the number on the chip and the number in the sentence would
 * be the same string by construction, so "they are equal" would hold for every
 * possible implementation, including a wrong one.
 *
 * **THAT ENFORCEMENT WAS MISSING, and it was missing in the place this file's
 * own header describes.** The first version promised the rule above and enforced
 * it only on the clipping path: `clipHeadline` returns any headline of 12 words or
 * fewer unchanged, so those rows were never compared against `headline.figures`
 * at all. CodeRabbit caught it against the vendored copy in F1
 * (`F1_Predictor#31`, comment `4172144009`) — "Backed by 61% of 30 picks" beside a
 * drawn 74%. That is the defect the split `headline` exists to make impossible,
 * reintroduced through the one branch that did not look. The test that should
 * have caught it did the opposite: it rendered a deliberately mismatched headline
 * and asserted the bar came out at 73.8%, which recorded the defect as expected
 * behaviour. A test that proves a thing is broken is not the same as a test that
 * catches it breaking, and this one was the former.
 *
 * ## What this component refuses, and why
 *
 * Three refusals, all of the same kind: a mistake a caller can make that would
 * otherwise reach the page as a *false statement* rather than as an error.
 *
 *  1. **A rate outside [0, 1]** (`SignalFigureError`). The "9600%" defect in
 *     this contract's clothes: `figures.rate` is a share, and a yardage figure
 *     read as one draws a bar that runs off its own track.
 *  2. **A figure the visual needs and the row does not carry**
 *     (`SignalFigureError`). A `reliability_bar` with no `figures.rate` would
 *     draw an empty track, and an empty track reads as 0%.
 *  3. **A visual this package cannot draw** (`UndrawableSignalVisualError`).
 *     Phase 1 draws `delta_chip` and `reliability_bar`. `absence_strip` (phase 2)
 *     and `projected_vs_actual` (phase 3) are refused rather than rendered
 *     without their marker, which is the one failure mode that both tests and
 *     screenshots miss.
 *  4. **A headline that does not state the figure its visual draws**
 *     (`HeadlineFigureMismatchError`). See `assertFigureIsStated`.
 *
 * ## What this component drops rather than refuses
 *
 * **A rate whose `n` is under the floor.** Spec §4: "Renders only at n >= 30;
 * below that the row says nothing, never a rate from a handful of games." So
 * the whole row goes, not just the rate — a headline reading "has been right
 * 74% of the time" with the bar and the `n` taken away is a worse row than no
 * row. `n: null` on a rate-drawing row is dropped for the same reason and is
 * **not** a special case of the arithmetic: there is no count, so there is
 * nothing that could have cleared a floor.
 *
 * ## Rejected: ranking and capping here
 *
 * Spec §2 puts the fixed rule in the signals layer — "a fixed rule (not the
 * model) ranks them by `strength` and keeps the top 2-3" — and the plan puts it
 * on the endpoint (`GET /signals/{game_id}` returns the top 3 by `strength`).
 * `PicksList` sets the opposite precedent and caps three rows per category
 * inside the component, because that is where it draws the rank numbers and
 * where a caller had already broken the rule. Neither transfers here. A
 * `SignalRows` row draws no position, so a cap in this component would silently
 * drop a row the endpoint ranked into its top three for a reason this component
 * cannot see, and a second copy of the ranking rule is a second rule — the
 * argument `scripts/ui-package.mjs` is written on. Rows render in the order they
 * arrive, and a test pins that.
 */
import { useId, useState } from "react";
import { kickoff, pct, signed } from "../fmt";

/** Spec §3's `kind`. All four are in the contract from phase 1; see
 *  `UndrawableSignalVisualError` for which of them this package can draw. */
export type SignalKind = "trust" | "absence" | "line_gap" | "post_game";

/** Spec §3's `visual`. */
export type SignalVisual = "delta_chip" | "reliability_bar" | "absence_strip" | "projected_vs_actual";

/**
 * Spec §3, field for field. The three places §3 leaves a shape open — `sport`,
 * `headline` and `as_of` — are resolved here, and the resolution is stated
 * rather than smuggled:
 *
 *  - `sport` and `as_of` are `string`. §3 writes `source: str, as_of: iso`, so
 *    `as_of` is an ISO-8601 string by the spec's own notation and is put into
 *    reader's words by `fmt.kickoff`, which every other date on these sites
 *    goes through. A template-literal type (`${number}-${number}-…`) was
 *    considered and rejected: it rejects dates that are valid and accepts
 *    strings that are not, because it never parses.
 *  - `headline` is `{ text, figures }`. §3 writes `headline: {figures...}`, an
 *    open object; the two halves are split for the reason in the file header.
 *  - `figures` is `Record<string, number>` rather than a union of per-visual
 *    shapes, because §3 leaves the figure names open and a caller is allowed to
 *    carry more than its visual draws. `signalFigure` below is what reads them,
 *    and it names the one it needs rather than reaching for whatever is there.
 */
export type Signal = {
  kind: SignalKind;
  sport: string;
  game_id: string;
  /** The one line, and the figures the visual draws. See the file header. */
  headline: { text: string; figures: Record<string, number> };
  /** Sample size where this signal states a rate; `null` where it states none. */
  n: number | null;
  /** Where the claim came from, in the adapter's own words. Rendered verbatim. */
  source: string;
  /** ISO 8601. "Attributable and dated" is §3's phrase, so a row carries both
   *  or neither; `source` is rendered in words rather than as a machine key. */
  as_of: string;
  /** How much the reader should care, 0..1. Set and tested by the adapter. */
  strength: number;
  /** False only for `post_game` (spec §3). */
  pre_kickoff_only: boolean;
  visual: SignalVisual;
};

/**
 * The floor on a rate's sample size.
 *
 * **This identifier is new here.** `TRUST_MIN_N` appears nowhere in this
 * repository: the plan names it as a constant each *sport adapter* declares in
 * Python (`docs/superpowers/plans/2026-10-02-fixture-signals-phases-1-4.md`,
 * "Constants per sport: `TRUST_MIN_N = 30`"), and no `Signal` code exists for
 * it to live in yet. So the number is named for where it comes from rather than
 * for what the plan calls it: spec §4's "`trust` … Renders only at n >= 30", and
 * spec §9 decision 3 ("trust needs n >= 30", reversible by "one constant per
 * sport").
 *
 * It lives in the shared component and not only in each adapter because the rule
 * is about what may be **drawn**, not about what may be computed. An adapter can
 * honestly hold a bucket of 12 games and report it; what it may not do is put a
 * rate built from it on the page. This is the last place a renderer can say no,
 * which is the argument `PicksList` makes with `OutPlayerInRankingError`.
 */
export const SPEC_MIN_N = 30;

/** Spec §6: "Each row has one line (<= 12 words)". */
export const MAX_HEADLINE_WORDS = 12;

/**
 * Which visual draws a RATE, and therefore which one the floor gates.
 *
 * Keyed by `visual`, not by `kind`, and that is the load-bearing choice: the
 * question the floor answers is "is a rate on the page", and `visual` is what
 * decides that. `kind` is the adapter's name for the finding and this component
 * cannot see how an adapter chose to draw it. A `delta_chip` row carries
 * magnitudes — a model's margin and a quoted line, both from this game — so the
 * historical disagreement count `n` is not what licenses them, and a chip row
 * with `n: null` is a complete row.
 *
 * No default branch, per `FactorList.MARK`: a fifth visual fails to compile
 * rather than falling through to "does not draw a rate", which is the reading
 * that would put a rate on the page unexamined.
 */
const DRAWS_A_RATE: Record<SignalVisual, boolean> = {
  reliability_bar: true,
  delta_chip: false,
  absence_strip: false,
  projected_vs_actual: false,
};

/**
 * The figure a visual names for itself, or `null` when this package draws no
 * marker for it at all.
 *
 * `delta_chip` draws the disagreement in the sport's own units; `reliability_bar`
 * draws the share of picks that hit. The two phase-2/3 visuals are `null` rather
 * than carrying a placeholder figure, because that is what makes "not drawable
 * yet" a fact this file can test rather than a comment. One map, so a caller
 * cannot reach a figure for a visual that has none.
 */
const FIGURE: Record<SignalVisual, "gap" | "rate" | null> = {
  delta_chip: "gap",
  reliability_bar: "rate",
  absence_strip: null,
  projected_vs_actual: null,
};

/**
 * A number a writer put in a headline, read out of the text.
 *
 * `cmp` is the boundary the writer stated (`<` or `>`) or `null` for a plain
 * number; `sign` is `+1`, `-1` or `0` for unsigned. Both are kept because both
 * carry meaning a renderer must not flatten — see `denotes`.
 */
type StatedFigure = { value: number; pct: boolean; cmp: "<" | ">" | null; sign: number };

/** The rate a bar row is about, in percent. Carried alongside `StatedFigure`
 *  because a drawn `<1%` needs BOTH its bound (what the reader sees) and the rate
 *  behind it (0.4) to judge a stated inequality: "is this claim true of the rate
 *  the bar draws" is a different question from "does this bound match that
 *  bound", and only the first one is the property the file cares about. */
type DrawnBar = StatedFigure & { rate: number };

/**
 * Every number in a piece of text, with its boundary and sign.
 *
 * **Matched with a scanner, not by splitting on whitespace and stripping
 * punctuation off each word.** Three CodeRabbit Majors on this rule, in order,
 * and each was the same mistake — a word was treated as the unit of a figure:
 *
 *  - `4171835614`: substring matching let `1%` satisfy `61%`.
 *  - `4171902960` / `4171924918`: normalising one side only let `<1%` stop
 *    matching itself.
 *  - `4172215626`: normalising BOTH sides discarded the boundary symbol, so a
 *    headline saying `1%` passed a row the chip labels `<1%` — which asserts a
 *    rate the bar does not draw.
 *  - `4172215627`: whitespace splitting left `74%—based` as one token, so a
 *    headline that plainly states `74%` was refused.
 *
 * Stripping punctuation off words cannot fix the last two and causes the third:
 * the boundary and the sign are *inside* the number, and a figure can be glued to
 * a word by an em dash without either of them being altered. So the text is
 * scanned for numbers and each is read whole, and the unit of comparison is the
 * NUMBER rather than the word. `1%` can no longer satisfy `61%` because the
 * scanner yields `61`, not `1`.
 *
 * Non-figure text is simply not matched and is therefore ignored, which is what
 * lets "Right 74%—based on 42 games" pass and "42 games" alone not count.
 *
 * **The sign is `+`, U+2212 or a hyphen-minus, and it is glued to its digits.**
 * An en dash and an em dash are typography, not arithmetic: with either in the
 * class, "a 5—3 record" parsed as a negative three and "a — 42 point swing"
 * parsed as −42, so a correct headline was refused because the figure it plainly
 * stated had been read with a sign nobody wrote. Removing them costs nothing —
 * no writer types an em dash to mean minus — and `\s*` between sign and digits
 * goes with them, because a spaced hyphen is a dash in prose ("Right — 74%"),
 * not a negative. The remaining three marks are unambiguous and always sit
 * against the number: `+1.6`, `−1.6`, `-1.6`.
 */
const STATED_FIGURE =
  /([<>])\s*([+−-]?)(\d+(?:\.\d+)?)\s*(%?)|([+−-]?)(\d+(?:\.\d+)?)\s*(%?)/g;

/** Every number in `text`, in order. */
function statedFigures(text: string): StatedFigure[] {
  const found: StatedFigure[] = [];
  for (const m of text.matchAll(STATED_FIGURE)) {
    // The group INDICES DIFFER per branch, and both branches capture a group 4:
    //
    //   branch 1 (has a comparator): m1=cmp  m2=sign  m3=digits  m4=%
    //   branch 2 (no comparator):    m5=sign m6=digits  m7=%
    //
    // Four groups each, so the two branches read DIFFERENT indices, and the
    // discriminator is `m[1] !== undefined`: a group that did not participate is
    // `undefined`, while a group that participated and matched an empty string is
    // `""`. Getting those indices wrong is not a subtle degradation — the first
    // version of this scanner read branch 2's digits from the group that holds
    // the sign, so `stated.value` was `Number("")` on every plain number and no
    // headline could state anything. Every case of the
    // headline-states-figure test failed at once, which is what made it obvious
    // rather than a rounding bug found in production.
    //
    // `??` is wrong here for the same reason and it is worth stating once: it
    // cannot tell "did not participate" from "participated and is empty", so it
    // reads the empty string from whichever branch lost.
    const withCmp = m[1] !== undefined;
    const cmp = (withCmp ? m[1] : null) as "<" | ">" | null;
    const raw = withCmp ? (m[2] ?? "") : (m[5] ?? "");
    const digits = (withCmp ? m[3] : m[6]) as string;
    const isPct = (withCmp ? m[4] : m[7]) === "%";
    found.push({
      value: Number(digits),
      pct: isPct,
      cmp,
      // A U+2212 MINUS, an en dash and a hyphen are all "negative" to a writer;
      // `fmt.signed` uses U+2212, and a headline may type any of the three.
      sign: /[+−–—-]/.test(raw) ? (raw === "+" ? 1 : -1) : 0,
    });
  }
  return found;
}

/**
 * The figure a visual draws, as the number a reader compares against.
 *
 * A rate is compared in PERCENT because that is how it is displayed: `0.738`
 * draws as `74%`, so a headline saying `74%` states it and one saying `0.738`
 * does not. A gap is compared as drawn, in the sport's own units, and its SIGN is
 * part of the figure — the chip reads `+1.6`, so `−1.6` is a different claim.
 */
function drawnAs(signal: Signal, figure: number): DrawnBar {
  if (signal.visual !== "reliability_bar") {
    // A gap's magnitude is its ABSOLUTE value and its sign is separate, because
    // "wants 1.6 fewer" states −1.6 with no minus sign in front of the number.
    // Comparing magnitudes as signed would refuse that, and the sign check below
    // is what compares direction.
    return {
      value: Math.abs(figure),
      pct: false,
      cmp: null,
      sign: figure > 0 ? 1 : figure < 0 ? -1 : 0,
      // Unused for a gap — it has no rate and no bound — but present so
      // `denotes` reads one shape rather than narrowing on `visual`.
      rate: Math.abs(figure),
    };
  }
  // A rate's own display carries the boundary: `pct` prints `<1%` and `>99%` at
  // its edges, and those are the symbols the reader sees — so the comparator AND
  // its bound are read back out of `pct`'s output rather than derived from the
  // value. That matters at the low edge: `pct(0.004)` prints `<1%`, where the bound
  // is 1 and the value is 0.4. Computing the bound as `Math.round(0.4)` gives 0,
  // which would refuse the headline's own `<1%` — the figure failing to match
  // itself, the same class as 4171902960, one level in. `pct` decides what the page
  // says, so reading it is the only source that cannot disagree with the page.
  const shown = pct(figure);
  const cmp = shown.startsWith("<") ? ("<" as const) : shown.startsWith(">") ? (">" as const) : null;
  const bound = cmp === null ? null : Number(shown.replace(/^[<>]/, "").replace("%", ""));
  // `rate` is the rate itself in percent, ALWAYS present for a bar row. `value` is
  // the BOUND when the display is an inequality and the percent otherwise, and
  // having both is what lets a stated inequality be evaluated rather than
  // pattern-matched — see `denotes`.
  return { value: cmp === null ? figure * 100 : (bound as number), pct: true, cmp, sign: 0, rate: figure * 100 };
}

/**
 * Does this stated number denote the drawn figure?
 *
 * Three conditions, each from a finding that a looser rule got wrong:
 *
 *  - **Magnitude, at the display's own rounding, by EQUALITY.** A rate is matched
 *    against the whole percent `pct` PRINTS and, where that percent was rounded up
 *    off a real fraction, against the whole percent below it — because at `0.738`
 *    the chip says 74% and a writer may say 73%, both being the same rate at the
 *    precision the page shows. Those candidates are read out of `pct`'s output and
 *    nothing else, and there is no tolerance on either side of the comparison: the
 *    candidates are already at the display's granularity, so a stated figure
 *    between them is a figure the bar does not draw. See the branch for why
 *    floating-point arithmetic cannot supply them and why `74.4%` is not a near
 *    miss of `74%`.
 *  - **A stated boundary must be TRUE OF THE DRAWN RATE.** The chip's `<1%` and
 *    `>99%` are not decoration: `<1%` means *less than* one percent, so a
 *    headline claiming `1%` states a rate the bar does not draw and is refused
 *    (4172215626). A boundaryless `0%` is consistent with `<1%` and is accepted.
 *    **The test is truth, not equality with the drawn bound** — a rate of 0.004
 *    drawn as `<1%` also makes `<5%` and `<0.5%` true, and both are accepted.
 *    Requiring the bounds to be *equal* (review comment 4172902588) would reject
 *    those two true claims, which trades a false positive for a false negative:
 *    the defect this file exists to prevent is a headline stating something
 *    untrue of the drawn figure, and an over-tight rule manufactures exactly that
 *    by rejecting what is merely more precise than the display's granularity.
 *    The claims that genuinely cannot be true of a share — `<0%` and `>100%` —
 *    are refused earlier on the domain, where `drawn.pct` is known, so they never
 *    reach the comparator branches at all.
 *  - **A stated sign must agree with the drawn one, for a gap.** The chip reads
 *    `+1.6`; a headline saying `−1.6` is claiming the opposite direction and is
 *    refused (4172215630). An UNSIGNED `1.6` is accepted, because a chip figure
 *    is routinely stated without its sign in prose and that is not a
 *    contradiction.
 */
function denotes(drawn: DrawnBar, stated: StatedFigure): boolean {
  // A percent and a bare number are different quantities. "Right 74%" does not
  // state a gap of 74, and "Wants 1.6 more" does not state a rate.
  if (drawn.pct !== stated.pct) return false;

  // A RATE is a share, so it is never negative. A headline stating `<−5%` or
  // `−0%` is describing a rate that cannot exist, and the inequality branches
  // below read `stated.value` without its sign — which is why this is checked
  // here rather than folded into them.
  if (drawn.pct && stated.sign < 0) return false;
  // Nor above 100%. `<1%`'s mirror image, and the same class as the bound cases.
  if (drawn.pct && stated.value > 100) return false;

  if (drawn.cmp !== null) {
    // **A bounded rate is EVALUATED, not pattern-matched.** The question is
    // whether the headline's claim is TRUE OF THE RATE THE BAR DRAWS, and the bar
    // draws `drawn.rate` — which is why `DrawnBar` carries the rate and not only
    // the bound. Three earlier shapes of this rule compared the drawn bound with
    // the stated one, and each admitted a claim that is false of the rate:
    //
    //  - value-inside-bound accepted `<0%` under `<1%` (0 is inside 1) and
    //    `>100%` under `>99%` (100 is outside 99, read as "more"). Both are
    //    impossible for a share: a rate is never negative and never above 100%
    //    (CodeRabbit review comment 4172902588).
    //  - accepting the bound with the same direction accepted `<0%`, because
    //    `<` matched `<`.
    //  - accepting any bound that was not WEAKER accepted `<50%` under `<1%`,
    //    which is a different rate entirely.
    //
    // So the stated figure is checked as a claim:
    //
    //  - **with the drawn comparator** — `<1%` — it must be TRUE of the rate,
    //    which for a rate of 0.4 means the bound must exceed it.
    //  - **with the other comparator** — `>0%` under a `<1%` bar — the rate is not
    //    above the stated bound, so the claim is false. A share is never above
    //    100%, which is why `>100%` is refused here rather than by a special case.
    //  - **unsigned** — it must be inside the drawn bound at display precision,
    //    which is what makes `0%` acceptable under `<1%` and `1%` not.
    const rate = drawn.rate;
    if (stated.cmp === drawn.cmp) {
      const holds = drawn.cmp === "<" ? rate < stated.value : rate > stated.value;
      if (!holds) return false;
    } else if (stated.cmp !== null) {
      // The opposite inequality, evaluated against the same rate.
      const holds = stated.cmp === "<" ? rate < stated.value : rate > stated.value;
      if (!holds) return false;
    } else {
      const inside = drawn.cmp === "<" ? stated.value < drawn.value : stated.value > drawn.value;
      if (!inside) return false;
    }
  } else if (drawn.pct) {
    // An unbounded RATE is compared at the DISPLAY's own precision, and `pct`
    // prints whole percents — so one whole percent is the granularity and the
    // stated figure must EQUAL one of the figures the bar can be read as.
    //
    // **The candidates come from the STRING THE PAGE PRINTS, not from float
    // arithmetic on the value.** `0.29 * 100` is 28.999999999999996, and
    // flooring that offered 28 as a figure the bar does not draw — a headline
    // saying `28%` was accepted beside a bar reading `29%`. The floor half is the
    // same arithmetic reading the wrong thing: 73.8 genuinely floors to 73, but
    // 28.999999999999996 is 29.000000 to any decimal the reader could write, so
    // its floor is a number no writer was ever stating. `pct(0.29)` is what the
    // page prints and it resolves the noise — `Math.round` sends 28.999999999999996
    // up to 29 — which is the whole reason `drawnAs` reads the comparator and the
    // bound back out of `pct`'s output rather than deriving them. The rounded
    // candidate is therefore READ OUT OF `pct`, and re-deriving it as
    // `Math.round(drawn.value)` would put this defect straight back on the next
    // rate that lands on a representable boundary.
    const rendered = Number(pct(drawn.rate / 100).replace("%", ""));
    //
    // **The floored candidate is reachable only where there is a REAL fraction to
    // truncate.** `pct` rounds 73.8 up to 74 and a writer truncating that same
    // rate may say 73% — the same rate at the precision the page shows, and the
    // one deliberate half of this rule. But when the value IS the rendered whole
    // percent there is nothing below it to state, so the candidate is not offered
    // at all. The test is `toFixed` rather than `Math.floor`, for the reason the
    // gap branch below spells out: `toFixed` is what the formatter uses to draw,
    // and it is what tells 29.000000 apart from 28.8. Read through `toFixed` the
    // floor is `73`, so the candidates are exactly `{74, 73}`.
    const truncated = Math.floor(Number(drawn.rate.toFixed(2)));
    const magnitudes = truncated < rendered ? [rendered, truncated] : [rendered];
    //
    // **EQUALITY, never a tolerance.** `Math.abs(m - stated.value) < 0.5` made the
    // rule half a percentage point wide, so at a rate of 0.738 — a bar reading
    // `74%` — a headline saying `74.4%` passed: a different number wearing one
    // decimal. That is the same defect as 4172337758 in the gap branch, in the
    // other direction, and it fails in the one direction this component cannot
    // afford: the reader is not told a wrong headline is wrong. The candidates are
    // already at the display's granularity, so the comparison needs no slack —
    // anything within half a point of them is a figure the bar does not draw.
    if (!magnitudes.includes(stated.value)) return false;
  } else {
    // A GAP is compared at ITS OWN precision, which is one decimal: `fmt.signed`
    // prints `+1.6` and never `+1.62`. When both paths shared the rate path's
    // whole-number tolerance this accepted `2` and `1.2` for a chip drawing `+1.6`
    // (CodeRabbit review comment 4172337758) — the same defect as 4172215630, in
    // the other direction: not the wrong sign but the wrong magnitude, and a
    // headline claiming a 2.0-point gap beside a chip reading 1.6 is a wrong number
    // on the page in exactly the way this file exists to prevent. Neither branch
    // carries a tolerance now; each states the display's own figures and compares
    // for equality, which is the same rule at two granularities.
    //
    // **The granularity is the DISPLAY's, never a fudge factor.** It is
    // derived from `fmt.signed` rather than written as `0.05`, because the rule
    // being enforced is "the words state the figure the chip draws" and the chip's
    // precision is defined by the formatter — a constant here would drift the day
    // `signed` changes.
    // **Both sides go through `toFixed`, because `toFixed` is what `fmt.signed`
    // uses to draw the chip**, AND the stated value must ROUND-TRIP at that
    // precision. Two separate requirements, and dropping the second is the trap
    // the suite already guards: rounding both sides alone makes `1.65` beside a
    // `+1.6` chip compare as `1.6 === 1.6` and be ACCEPTED, when `1.65` is a
    // different figure wearing more decimals than the page shows. The
    // round-trip is what keeps it a different figure.
    //
    // Rounding with `Math.round` instead was the bug: `signed(1.65)` prints
    // `+1.6` (1.65 sits below 1.65 in binary) while `Math.round(1.65 * 10)` is
    // 17, so a headline copying the chip's own `+1.6` was refused for failing to
    // match a chip reading `+1.6` — the figure disagreeing with itself.
    const decimals = (signed(drawn.value).split(".")[1] ?? "").length;
    const atDisplay = Number(stated.value.toFixed(decimals));
    if (atDisplay !== stated.value) return false;
    if (atDisplay !== Number(signed(drawn.value))) return false;
  }

  // A stated comparator on an UNBOUNDED drawn rate is refused outright: `<2` on a
  // chip that draws `2` narrows the claim, and this component cannot tell a
  // legitimate hedge from a wrong number. The bounded branch above has already
  // handled the other direction, by evaluating the claim against the rate rather
  // than requiring the symbols to match — which is why this is not simply
  // `stated.cmp !== drawn.cmp`: that refused `>0%` under a `<1%` bar, which is a
  // true statement about a rate of 0.4.
  if (drawn.cmp === null && stated.cmp !== null) return false;

  // A gap's sign is the direction; a rate's is not (a rate has no direction).
  if (!drawn.pct && stated.sign !== 0 && stated.sign !== drawn.sign) return false;
  return true;
}

/** Does this headline state this figure, in any of the words a writer would use?
 *
 * The unit of comparison is the NUMBER, not the word — see `STATED_FIGURE`.
 */
export function headlineStatesFigure(text: string, signal: Signal, figure: number): boolean {
  const drawn = drawnAs(signal, figure);
  return statedFigures(text).some((stated) => denotes(drawn, stated));
}

/**
 * Does the LAST figure in this text state the drawn one?
 *
 * A separate question from `headlineStatesFigure`, and the clipper needs this
 * one. It walks a headline word by word asking whether a figure ENDS at this word
 * (a comparator and its number may straddle a space — `below < 1%` — so the span
 * it offers can be more than one word). If the span carries a figure anywhere,
 * "any figure in the span" would keep `b c 74%` whole and drag two ordinary words
 * in with it; the figure the span ends with is the one that starts at it.
 *
 * Without the distinction the two halves of this rule disagree: the validator
 * accepts a headline because a figure appears SOMEWHERE in it, and the clipper
 * then protects far more than that figure, which is prose the reader did not ask
 * to keep.
 */
/**
 * Where the drawn figure sits in this text, as character offsets into it.
 *
 * **A RANGE, and that is the third shape this question has taken.** It was first
 * "which words state the figure" (a list), then "does this span state it" (a
 * predicate), and both are the wrong interface: a predicate cannot say *which
 * words* to keep, so the clipper had to re-derive it by asking about prefixes,
 * and asking "does this span END with the figure" matches `b c 74%` — three words
 * for a figure that is one (CodeRabbit review comment 4172337760 and the two
 * regressions it produced, where the figure came out printed twice).
 *
 * A range says it exactly: the clipper keeps the words this range covers and
 * nothing else. `below < 1%` is one range across two words because the scanner
 * read it as one figure; `1.6` in `1.6 more` is one word because it is one.
 *
 * The FIRST figure in the text is the one returned, so a headline that states it
 * twice protects the first occurrence — which is where the reader reads it, and
 * keeping the second would cost prose for nothing.
 */
export function figureRange(
  headline: string,
  signal: Signal,
  figure: number,
): { start: number; end: number } | null {
  // **Offsets are into the TRIMMED headline**, because that is the string
  // `clipHeadline` walks. The first version took the caller's text and returned
  // offsets into it, and `clipHeadline` applied them to `text.trim()` — so a
  // headline with a leading space had every offset shifted by one and the range
  // could protect `tail` instead of `74%`, silently clipping away the figure the
  // validator had just accepted (CodeRabbit review comment 4172902590).
  //
  // Trimming HERE is what makes the two agree, and `clipHeadline` trims the same
  // way for the same reason. The alternative — passing the untrimmed string in and
  // having the clipper re-derive the offset — is a second place to get it wrong.
  const text = headline.trim();
  const drawn = drawnAs(signal, figure);
  const matches = [...text.matchAll(STATED_FIGURE)];
  for (let i = 0; i < matches.length; i++) {
    if (denotes(drawn, statedFigures(text)[i])) {
      return { start: matches[i].index, end: matches[i].index + matches[i][0].length };
    }
  }
  return null;
}

/** A visual this package cannot draw. */
export class UndrawableSignalVisualError extends Error {
  constructor(gameId: string, visual: string, kind: string) {
    super(
      `SignalRows: signal ${gameId} (${kind}) asks for visual "${visual}", which this package does ` +
        `not draw. Rendering the row without its marker would put a headline on the page with the ` +
        `figure beside it missing, so it is refused instead. Drawable today: delta_chip and ` +
        `reliability_bar. absence_strip arrives with the absence signal, projected_vs_actual with ` +
        `post_game.`,
    );
    this.name = "UndrawableSignalVisualError";
  }
}

/** A figure that cannot honestly draw its visual, or is not there at all. */
export class SignalFigureError extends Error {
  constructor(gameId: string, kind: string, visual: string, name: string, detail: string) {
    super(
      `SignalRows: signal ${gameId} (${kind}) draws "${visual}" from headline.figures.${name}, ` +
        `${detail}.`,
    );
    this.name = "SignalFigureError";
  }
}

/**
 * The figure a visual draws, or a throw naming what was wrong with it.
 *
 * Checked before anything renders, so a caller that made the mistake gets a
 * named error naming the game and the figure instead of a bar running off its
 * own track — the same ordering `PicksList.checkRow` uses.
 */
export function signalFigure(signal: Signal, visual: SignalVisual): number {
  const name = FIGURE[visual];
  if (name === null) throw new UndrawableSignalVisualError(signal.game_id, visual, signal.kind);
  const value = signal.headline.figures[name];
  if (value === undefined) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      "which the signal does not carry, and an empty visual reads as zero",
    );
  }
  if (!Number.isFinite(value)) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      `which is not a finite number (got ${value})`,
    );
  }
  // Only a rate has a range. `gap` is a margin in the sport's own units and is
  // checked as a magnitude: a yardage figure is a perfectly good gap, and the
  // "9600%" defect is a share read off the wrong field, not a large number.
  if (name === "rate" && (value < 0 || value > 1)) {
    throw new SignalFigureError(
      signal.game_id, signal.kind, visual, name,
      `which must be a share in [0,1] — got ${value}. A yardage figure read as a hit rate is the ` +
        `defect this guard exists for`,
    );
  }
  return value;
}

/**
 * THE floor a page will not go below, whatever a caller asks for.
 *
 * One place, and it is deliberately *not* the component: `rateIsDrawable` is
 * exported, so a site gating its own copy of a list through it must get the same
 * answer `SignalRows` would give. Clamping in the component as well would leave
 * two rules to keep in step — the thing `scripts/ui-package.mjs` is written
 * about. A caller may therefore **raise** the floor and can never lower it.
 */
const floorOf = (minN: number): number => Math.max(SPEC_MIN_N, minN);

/**
 * May this signal put a rate on the page?
 *
 * Three ways to answer no, and all three have been a real shape of the mistake:
 * the visual draws no rate (nothing to gate); the row states no `n`; the `n` is
 * under the floor. A `delta_chip` answers true whatever its `n`, because the
 * figures it draws are this game's own.
 */
export function rateIsDrawable(signal: Signal, minN: number = SPEC_MIN_N): boolean {
  if (!DRAWS_A_RATE[signal.visual]) return true;
  const n = signal.n;
  // `Number.isInteger`, not `n >= floor`: an adapter that put a float where a
  // count belongs has not reported a sample size, and `29.5 >= 30` is false by
  // luck rather than by rule.
  return Number.isInteger(n) && (n as number) >= floorOf(minN);
}

/** A headline that does not state the figure its own row draws. */
export class HeadlineFigureMismatchError extends Error {
  constructor(gameId: string, kind: string, visual: string, name: string, drawn: string, stated: string) {
    super(
      `SignalRows: signal ${gameId} (${kind}) draws "${visual}" from headline.figures.${name} as ` +
        `${drawn}, but its headline never states that figure — it reads "${stated}". A figure on ` +
        `the page that the row's own words do not carry is the one thing this component exists to ` +
        `prevent, so the row is refused rather than rendered beside words that disagree with it.`,
    );
    this.name = "HeadlineFigureMismatchError";
  }
}

/**
 * THE RULE, enforced on every path: the figure this visual draws must appear in
 * this headline's words.
 *
 * **WHY THIS THROWS RATHER THAN DROPPING THE HEADLINE.** Both were considered and
 * the cost is stated here because it is real: a throw takes the whole list down,
 * so one adapter that mis-states a figure takes out every other signal on that
 * fixture's page.
 *
 * It throws anyway, for three reasons, and the third is the one that settles it.
 *
 *  1. **The precedent is two-for-two in this package.** `RowKindMismatchError` and
 *     `OutPlayerInRankingError` in `PicksList.tsx` both throw, for the same shape
 *     of mistake: a caller passed something the component must not draw. A third
 *     component answering the same question differently would mean a caller had
 *     to learn which rule applied where.
 *  2. **The bug is in the CALLER, and the caller has tests.** An adapter that
 *     writes "61%" beside `figures.rate = 0.74` is broken, and it is broken in
 *     that adapter's own suite before it reaches a reader. Throwing is what puts
 *     the failure in front of whoever can fix it, which is the adapter's author,
 *     and it names the game and both readings.
 *  3. **Dropping the headline does not degrade quietly — it degrades to the very
 *     thing the rule forbids.** A row whose bar draws 74% with no words beside it
 *     is a bare number with no claim attached and no sample size in words. The
 *     reader cannot tell a 74% from a 61% because neither is on the page. That is
 *     not a lesser version of the defect; it is the same reader-facing outcome
 *     with the part that made the number checkable removed. The only reason
 *     dropping would be better is that the page survives — and a page that loses
 *     a row because an author made a typo is a page that ships the typo's
 *     consequence silently, which is the failure mode every refusal in this file
 *     exists to avoid.
 *
 * **A throw in a shared UI is recoverable**, because this family already renders
 * failures: `States.tsx`'s `ErrorState` is a `role="alert"` with a retry, and
 * `ExplainerPanel` uses it. A site that would rather lose the row can catch this
 * at its own boundary; what it must not do is catch it and render the figure
 * beside words that disagree with it.
 *
 * **What it cannot catch, stated rather than implied.** This checks that the
 * figure appears, not that the sentence is *true*. A headline saying "the model
 * has been right 74% of the time" with `figures.rate = 0.74` passes, and that
 * sentence is the adapter's claim about what the number means. Nothing in a
 * renderer can check that; it is the adapter's own test, and it is the reason
 * `figures` and `text` are both in the payload rather than one derived from the
 * other.
 */
export function assertFigureIsStated(signal: Signal): void {
  const figure = signalFigure(signal, signal.visual);
  if (!headlineStatesFigure(signal.headline.text, signal, figure)) {
    const name = FIGURE[signal.visual]!;
    const drawn = signal.visual === "reliability_bar" ? pct(figure) : signed(figure);
    throw new HeadlineFigureMismatchError(
      signal.game_id, signal.kind, signal.visual, name, drawn, signal.headline.text,
    );
  }
}

/**
 * Is this signal a row at all?
 *
 * Four of the spec's own rules, three of them "no data, no row" and one a refusal:
 * a blank headline says nothing; a rate under the floor says nothing (spec §4); a
 * visual this package cannot draw is not a row **this** component is willing to
 * draw, so it throws rather than answering false; and a headline that does not
 * state the drawn figure throws rather than rendering a bare number with no words.
 *
 * **The order is load-bearing, and it is the floor interaction, settled
 * deliberately.** The floor is checked BEFORE the figure is validated, so a
 * below-floor rate is **dropped, never rejected**: the figure it carries is
 * suppressed and its headline goes with it, so there is nothing on the page for
 * the figure to contradict. Validating first would make every sub-floor bucket a
 * thrown error — a sport with twelve games of history could not render its panel
 * at all — which is spec §4's answer inverted: §4 says a sub-floor rate says
 * nothing, and "nothing" is not a crash. The two rules therefore never meet: a
 * dropped row is not a contradicted row, because it has no figure to state.
 *
 * The flip side is stated because it is a real limit: a headline that
 * mis-states a figure on a row that the floor drops anyway is **not** caught. That
 * is accepted — the row was going to be invisible, and a validation that fires on
 * invisible rows converts a suppressed row into a broken page for no reader-facing
 * gain. The adapter's own test is where that payload is caught.
 */
export function signalIsDrawn(signal: Signal, minN: number = SPEC_MIN_N): boolean {
  if (FIGURE[signal.visual] === null) {
    throw new UndrawableSignalVisualError(signal.game_id, signal.visual, signal.kind);
  }
  if (signal.headline.text.trim() === "") return false;
  if (!rateIsDrawable(signal, minN)) return false;
  assertFigureIsStated(signal);
  return true;
}

/**
 * Clip a headline to `max` words, and say whether it was.
 *
 * **The rule: count whitespace-delimited words, keep the first `max`, and mark
 * the row `data-clipped`.** A word is a whitespace-delimited token because that
 * is what fits on one line — "−4.5:" is one word to a reader and two to a
 * tokenizer that also splits on punctuation.
 *
 * **Why clipping and not refusing, when a wrong number is not acceptable.**
 * Because clipping removes *words* and never edits a *figure*. The figures this
 * component draws come from `headline.figures`, which clipping does not touch,
 * so a clipped row still shows exactly the numbers it claims and loses only
 * prose — and the sample size survives regardless, because `n` is on the evidence
 * line whether or not the headline is clipped. A refused row, by contrast, loses
 * its whole finding because an adapter miscounted a word, which is a worse
 * outcome than a shorter sentence and a `data-clipped="true"` in the DOM.
 *
 * **A drawn figure is PROTECTED, and this was a real defect until it was.**
 * The first version claimed a clip "cannot corrupt a figure" on the grounds that
 * the figures are not in the text this function rewrites. That was wrong, and
 * CodeRabbit review comment 4171773475 caught it: the headline *does* speak them
 * — that is the whole honesty rule above — and "was right 74% of the time" can
 * sit at word 14. Clipping to 12 then left a bar at 74% under a sentence that no
 * longer mentioned 74%: exactly the dishonesty the split `headline` exists to
 * make impossible, reintroduced by the cap.
 *
 * So the rule is now: **keep the first `max` ORDINARY words, plus every word the
 * drawn figure occupies, in document order.**
 *
 * **Protection is a RANGE of character offsets, supplied by the caller, and this
 * is the third interface this question has had.** That history is worth keeping
 * because each shape failed in the same direction — it could say WHETHER the
 * figure was stated but not WHICH WORDS to keep:
 *
 *  - a `protectedWords: string[]`, matched by stripping punctuation off each
 *    word. Substring matching let `1%` satisfy `61%` (4171835614); normalising one
 *    side only let `<1%` stop matching itself (4171902960, 4171924918); stripping
 *    both threw the boundary away so `1%` satisfied `<1%` (4172215626); and
 *    splitting on whitespace left `74%—based` unrecognised so a headline plainly
 *    stating `74%` was clipped away (4172215627). Root cause of all four: the unit
 *    was the WORD, and a figure is not a word.
 *  - a PREDICATE over spans of up to three words, which fixed the straddling case
 *    but could not say which words the figure covered — so the clipper asked about
 *    prefixes of itself and kept `b c 74%` whole (4172337760), and a scan that
 *    resumed one word at a time re-found the same figure and printed it twice.
 *
 * A range says exactly which characters are the figure, so the clipper keeps the
 * words those characters sit in and re-derives nothing. It is built by
 * `figureRange` from the same `denotes` the validator uses, which is what makes
 * the two halves of this rule one rule: the validator asks "does the headline
 * state the figure at all", the clipper asks "which words is it".
 *
 * **And the figure being protected must be one the headline actually stated.**
 * `signalIsDrawn` → `assertFigureIsStated` runs first, so the range is never null
 * for a row that reaches the screen: a row whose words omit the figure is refused
 * rather than clipped.
 *
 * **Where the cap sits.** The cap is on the sentence AROUND the figure, so a
 * clipped row can exceed `max` words, and a headline that names its figure on
 * every one of forty words stays forty words long. That is an adapter copy bug
 * with an adapter fix, and `data-clipped` is the signal for it. The words BESIDE a
 * protected figure are ordinary and spend the budget like any other — pinned by a
 * test, because an earlier interface swept them in.
 */
export function clipHeadline(
  text: string,
  max: number = MAX_HEADLINE_WORDS,
  keep?: { start: number; end: number } | null,
): { text: string; clipped: boolean } {
  const trimmed = text.trim();
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length <= max) return { text: trimmed, clipped: false };

  // Which words does the protected range touch? Character offsets become word
  // indices ONCE here, so the clipper never re-asks what the figure is — that was
  // the defect in the two previous interfaces, where it asked about prefixes of
  // itself and kept `b c 74%` whole for a figure that is one word.
  //
  // Overlap rather than containment, because the range can start INSIDE a word
  // when an adapter writes `74%—based`, and because a two-word figure
  // (`below < 1%`) must keep both of its words.
  const protectedWords = new Set<number>();
  if (keep) {
    // `trimmed` is what the offsets are into, so this walks that same string.
    let at = 0;
    for (let i = 0; i < words.length; i++) {
      const start = trimmed.indexOf(words[i], at);
      const end = start + words[i].length;
      if (start < keep.end && end > keep.start) protectedWords.add(i);
      at = end;
    }
  }

  const kept: string[] = [];
  let ordinary = 0;
  for (let i = 0; i < words.length; i++) {
    if (protectedWords.has(i)) {
      // A protected figure is kept wherever it is and does not spend the budget:
      // the cap is on the sentence AROUND the figure, not on the figure.
      kept.push(words[i]);
      continue;
    }
    if (ordinary < max) kept.push(words[i]);
    ordinary++;
  }
  return { text: kept.join(" "), clipped: ordinary > max };
}

/** The heading a list of rows carries. Plain words and no developer term:
 *  PRODUCT.md requires plain-language copy on public pages, so this is not
 *  "Signals". */
export const DEFAULT_SIGNAL_TITLE = "Context";

export interface SignalRowsProps {
  signals: Signal[];
  /** The heading the rows sit under. Overridable because a site may already
   *  have one in view, and two stacked headings is noise. */
  title?: string;
  /**
   * The floor, for a sport whose own buckets sit under the spec's.
   *
   * **It can only raise it**, because `floorOf` clamps it — the spec's 30 is
   * the floor the page will not go below, so a prop that lowered it would make
   * the guarantee this package exists to provide opt-out-able by the caller with
   * the most reason to want it out. Spec §9 decision 3 reverses that floor "by
   * one constant per sport", and a sport needing a *higher* bar is exactly what
   * this prop is for.
   */
  minN?: number;
}

/** One row. Not exported, so every path into it goes through `signalIsDrawn`
 *  first and no caller can mount a row the floor has not cleared. */
function SignalRow({ signal, rowId }: { signal: Signal; rowId: string }) {
  const [open, setOpen] = useState(false);
  const isBar = signal.visual === "reliability_bar";
  const n = signal.n;
  // The count as evidence words. Empty when the row states none, so a label can
  // never read "of null games"; the floor means a bar-drawing row always has one.
  const count = n === null ? "" : `n=${n}`;
  const figure = signalFigure(signal, signal.visual);
  // Clipped with the figure PROTECTED, and the protection is the SAME question
  // `assertFigureIsStated` just asked about the whole headline — asked again per
  // word, so a figure the clip would otherwise drop is kept. That function ran
  // first and refused the row if no word stated the figure, so at least one word
  // here is protected whenever this row reaches the screen.
  const { text, clipped } = clipHeadline(
    signal.headline.text,
    MAX_HEADLINE_WORDS,
    // The RANGE of the drawn figure, so the clipper keeps the words it covers and
    // re-derives nothing. `assertFigureIsStated` has already established that one
    // exists, so this is never null for a row that reaches the screen.
    figureRange(signal.headline.text, signal, figure),
  );

  return (
    <li
      data-testid="signal-row"
      data-row={rowId}
      data-kind={signal.kind}
      data-visual={signal.visual}
      className="flex min-w-0 flex-col gap-1.5 border-t border-pr-rule pt-3 first:border-t-0 first:pt-0"
    >
      <div className="flex min-w-0 items-baseline gap-2">
        {signal.visual === "delta_chip" && (
          // Neutral, and never `pr-win` / `pr-loss`. The spec's own case-8 mock
          // painted this chip green (`text-pr-win`), and a disagreement with the
          // line is not an outcome: green beside red would be two rows claiming
          // the model was right and wrong about the same kind of thing. The sign
          // carries the direction, which is also `FactorList`'s rule — a
          // direction is never carried by colour.
          <span
            data-testid="signal-delta"
            className="shrink-0 rounded-pr border border-pr-rule px-1.5 py-0.5 font-pr-display text-xs font-semibold tabular-nums text-pr-text"
          >
            {signed(figure)}
          </span>
        )}
        <p
          data-testid="signal-headline"
          data-clipped={clipped ? "true" : "false"}
          className="min-w-0 flex-1 text-sm font-medium leading-snug text-pr-text"
        >
          {text}
          {clipped && <span aria-hidden="true"> …</span>}
        </p>
      </div>

      {isBar && (
        // `role="img"` with the figures in words, as `RecordStrip` does and for
        // the same reason: a screen reader cannot see a track's width. It is not
        // `aria-hidden` the way `PicksList`'s bar is, because a headline may be
        // clipped (see `clipHeadline`) and this number has to survive that.
        <span
          role="img"
          aria-label={`${pct(figure)} right${count ? ` of ${String(n)} games` : ""}`}
          className="flex h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2"
        >
          <span data-testid="signal-bar-fill" className="h-full bg-pr-accent" style={{ width: `${figure * 100}%` }} />
        </span>
      )}

      {/* The evidence line, collapsed.
          *
          * A real `<button>`, not a `title` and not a hover-only line: a `title`
          * never appears on the device this is mostly read on and is not reliably
          * announced from a span, and evidence nobody can reach is not evidence.
          * No `aria-controls` — the family's other disclosures (`FactorList`,
          * `ProbabilityBar`) carry none either, `aria-expanded` is the part that
          * is load-bearing, and the element it would name does not exist while
          * the panel is closed.
          *
          * `n` and the date are ALWAYS visible and OUTSIDE the control, in the
          * panel's own small faint type rather than the button's uppercase
          * tracking. Two reasons, and the first is the whole one: `n` is the
          * sample size, so uppercasing it prints `N=61` and a reader takes that
          * for a name rather than a count — and a count is the one thing on this
          * row a sceptic needs without opening anything.
          *
          * **WHAT IS AMBIGUOUS IN THE SPEC, and the reading taken.** §6 asks for
          * "a collapsed evidence line (source, n, date)", and its parenthetical
          * names three fields while "collapsed" names a hidden state — so it
          * cannot mean all three are behind the control. This reads it as: the
          * line carries all three, and what collapses is the LONG one. `source`
          * is a free string from another repository ("stored pre-kickoff picks,
          * 0.5-0.6 bucket") and is the field that makes a row two or three lines
          * deep at 390px; `n` and a formatted date are short and cost one line.
          * So `n` and the date sit in the collapsed row and `source` is what the
          * button reveals. A reading where all three hide was rejected: it makes
          * the sample size unavailable without a pointer's worth of interaction
          * on the exact claim a reader most wants to check. */}
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <button
          type="button"
          data-testid="signal-evidence-toggle"
          aria-expanded={open}
          onClick={() => setOpen((was) => !was)}
          className="shrink-0 text-xs font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 hover:underline"
        >
          {open ? "Hide source" : "Source"}
        </button>
        <p data-testid="signal-evidence" className="min-w-0 text-xs leading-snug text-pr-text-faint">
          {[count, kickoff(signal.as_of)].filter(Boolean).join(" · ")}
          {open && signal.source ? ` · ${signal.source}` : ""}
        </p>
      </div>
    </li>
  );
}

export function SignalRows({ signals, title = DEFAULT_SIGNAL_TITLE, minN }: SignalRowsProps) {
  const headingId = useId();
  // The raw caller value; `rateIsDrawable` is what clamps it. See `minN`.
  const floor = minN ?? SPEC_MIN_N;
  // Every row is checked before any of them renders, so one bad row cannot
  // leave a half-drawn list on the page.
  const drawn = signals.filter((signal) => signalIsDrawn(signal, floor));

  // An empty list renders nothing at all: no heading, no panel, no "no signals",
  // no zero. Spec §2 — "No data, no row. No empty states, no 'unknown' rows, no
  // filler" — and a list whose every row was dropped for want of a sample is the
  // same empty list, so it renders nothing too.
  if (drawn.length === 0) return null;

  return (
    <section aria-labelledby={headingId} className="flex min-w-0 flex-col gap-3">
      {/* `<h3>`, the panel's own section level (`PanelHeading`), and not the
          `<p>` `PicksList` uses for its title: a `<p>` gives a screen reader no
          landmark to navigate to, and this list is the thing a reader of the
          facts block is looking for. */}
      <h3
        id={headingId}
        data-testid="signal-rows-title"
        className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        {title}
      </h3>
      <ul data-testid="signal-rows" className="flex min-w-0 flex-col gap-3">
        {/* The key is `game_id:kind:index`, and the index is in it on purpose.
         *
         * Spec §3 carries no row id, and this component does not add one: the
         * type is the contract and inventing a field no adapter has to populate
         * would be a field every real payload leaves undefined.
         *
         * `game_id:kind` alone is not a key. CodeRabbit review comment
         * 4171773477 is right that §3 does not make those unique, and two
         * signals sharing one is a real possibility (two `trust` buckets on one
         * game is plausible). React would then reuse one `SignalRow`'s
         * `useState` for both, which is the row's disclosure state showing on
         * the wrong signal.
         *
         * The index's own weakness is named here rather than glossed: a reorder
         * would carry the open disclosure with the position rather than the
         * signal. That is accepted because rows render in the order the endpoint
         * ranked them (see the file header), a reorder between two renders of
         * the same fixture is not a case this component produces, and the
         * alternative — a key of `(game_id, kind)` alone — is wrong NOW rather
         * than wrong in a case that has not happened. What would remove both
         * costs is an `id` in §3; that is a spec change, so it is reported
         * rather than taken here. */}
        {drawn.map((signal, i) => (
          <SignalRow
            key={`${signal.game_id}:${signal.kind}:${i}`}
            signal={signal}
            rowId={`${signal.game_id}:${signal.kind}:${i}`}
          />
        ))}
      </ul>
    </section>
  );
}