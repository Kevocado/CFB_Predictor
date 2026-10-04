// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * The model's own top calls, ranked per category.
 *
 * Three rules are enforced HERE rather than left to every caller, because each
 * one has already been broken at least once by a caller that was being careful:
 *
 *  1. **A row's visual must match what its number IS.** `kind: "probability"`
 *     draws a share bar and `kind: "projection"` draws a key number with a ±.
 *     Passing a yardage figure as a probability once rendered "9600%", so the
 *     two kinds have disjoint ranges and a value outside its own is refused by
 *     name rather than drawn.
 *  2. **An out player leaves the ranking entirely.** A row marked `out` is
 *     refused. There is no greyed-out row and no "flagged in place" state to
 *     opt into — the caller passes them in `out`, where they are shown once,
 *     below the lists, with the source and the date.
 *  3. **Three rows per category, and three is a ceiling.** Fewer when fewer
 *     exist. Nothing is padded to fill the list.
 *
 * Provenance is the caller's own words, verbatim. This component composes no
 * confidence language at all: a probability, a bucket context and a ± are all
 * it renders, so there is no sentence here that could claim an edge, a price
 * or a guarantee — no odds feed exists in any of these repos.
 *
 * And a fourth rule, which is not a guard against a careless caller but a
 * decision about what a row says:
 *
 *  4. **A row's `detail` is drawn only on a probability row whose detail adds
 *     a word the category heading does not already carry** — `rowShowsDetail`.
 *     See it for the rule, the two readings of "always" that were rejected, and
 *     the real `Sports_Predictor` payload that defeated the first version of
 *     this rule.
 */
import { pct, stat } from "../fmt";
import { StatusBadge } from "./StatusBadge";

/** A player out for this fixture, shown once below the lists and never ranked. */
export type OutPlayer = {
  name: string;
  team?: string;
  /** Where the out claim came from, in words. Never an id. */
  source: string;
  /** When it was known, in words. Never an epoch. */
  dated: string;
};

export type PickRow = {
  /** Stable per player+category; the tile's identity and the React key. */
  key: string;
  name: string;
  team?: string;
  /** The category, in words: "Anytime TD", "Rush yds".
   *
   *  Two incompatible jobs have shared this field. Historically it is the
   *  category restated — `Sports_Predictor`'s `positionCategories()` types it
   *  as `detail: "Pass yds"` under `category: "QB passing yards"` — and a
   *  restatement is drawn as nothing. Newer it is the CALL: `"Over 2.5"` with
   *  the probability in `value`, which is what makes NFL's QB passing-TD rows
   *  read `Over 2.5 · 64%` instead of a bare percentage with no line.
   *
   *  A component cannot tell those two apart from the string alone, so it does
   *  not try: `rowShowsDetail` keys on `kind`, which the payload already states.
   */
  detail: string;
  /** The model's own number for this category. Never derived here. */
  value: number;
  /** "probability" draws a share bar; "projection" draws a key number + ±. */
  kind: "probability" | "projection";
  /** Only for `kind: "projection"`: the error margin, rendered as "± N".
   *  Absent means there is no error estimate yet, which is said in words
   *  rather than drawn as "± 0" — a zero margin would be a claim. */
  margin?: number;
  /** What the rendered `detail` IS, in the caller's own words. Defaults to
   *  `DEFAULT_DETAIL_LABEL`. **Read only on a `kind: "probability"` row** — see
   *  `rowShowsDetail`. A projection row draws no qualifier at all, including one
   *  passed here, because a qualifier is a claim that the row states a call and
   *  a projection states a magnitude.
   *
   *  The qualifier is in the *visible* text, not in a `title` or a hidden
   *  accessible label, and that is the whole point of it. A detail like
   *  "Over 2.5" is the shape of a sportsbook price, and nothing on the page
   *  distinguishes one from the other: there is no odds feed in any of these
   *  repos, so a reader who assumes a price is reading a line the model chose.
   *  A `title` cannot carry that on the device this is mostly read on — it
   *  never appears on touch, and it is not reliably announced from a
   *  non-focusable span — so the provenance has to be words on the page.
   *
   *  "model call" is the default because it is the one noun this component
   *  can vouch for on a probability row: the model is calling that outcome to
   *  happen at that share, and `value` is documented above as the model's own
   *  number. A caller that knows its detail is a line says so
   *  (Sports_Predictor's NFL adapter passes `"model line"`, which is the
   *  plan's wording) — which is also why this is per row and not a prop on
   *  `PicksListProps`: a list can hold two kinds of row.
   */
  detailLabel?: string;
  /** Accepted for compatibility and NOT rendered. Kevin, 2026-10-01: the best calls are
   *  the player and the prediction, nothing else (no record, Brier, calibration or error
   *  prose per row). Callers may stop passing it. */
  provenance?: string;
  /** True when this player is out. If true the row must NOT be passed — see
   *  `out`. The field exists so the mistake can be caught by name instead of
   *  rendering a forbidden ranking. */
  out?: boolean;
};

export interface PicksListProps {
  title?: string;
  categories: { category: string; rows: PickRow[] }[];
  out?: OutPlayer[];
}

/** At most this many rows per category. A cap, never a quota. */
export const MAX_ROWS_PER_CATEGORY = 3;

/** The qualifier that says where a rendered `detail` came from. Exported so a
 *  site names this constant rather than retyping the string. */
export const DEFAULT_DETAIL_LABEL = "model call";

/**
 * Words, lowercased, with runs of punctuation split and the dot inside a number
 * kept, so `2.5` is one token rather than two.
 */
function words(text: string): string[] {
  return text.toLowerCase().split(/[^a-z0-9.]+/).filter(Boolean);
}

/**
 * Does this `detail` say anything its category heading does not already say?
 *
 * Token containment, not string equality: the harness already builds headings
 * like `"Rush yds — projections, not probabilities"`, and `"Rush yds" !==` that
 * string, so an equality test would print the detail three times on every card
 * and call the output a fix.
 *
 * **WHAT IT CANNOT SEE, because it is a string comparison and not a parser.**
 * It cannot tell an abbreviation from a new word. `detailAddsToHeading("Pass
 * yds", "QB passing yards")` is `true`, because `{pass, yds}` and
 * `{qb, passing, yards}` share no token — and that is a restatement, drawn.
 * `rowShowsDetail` below is what stops that row ever reaching this function; do
 * not widen its gate without reading this paragraph first.
 */
export function detailAddsToHeading(detail: string, category: string): boolean {
  const detailWords = words(detail);
  if (detailWords.length === 0) return false;
  const headingWords = new Set(words(category));
  return detailWords.some((word) => !headingWords.has(word));
}

/**
 * THE RULE. Does this row draw its `detail`, with a qualifier?
 *
 * Two conditions, and the first one is not optional.
 *
 * **1. `kind === "probability"`.** A call is a share — "this happens, at this
 * price of confidence" — and a projection is a magnitude: 284 yards, 10.2
 * rebounds. `kind` is the field this component already treats as the
 * authoritative statement of what a row's number IS, to the point of throwing
 * `RowKindMismatchError` at a row whose kind contradicts its value, so it is
 * the one thing here that cannot be a guess. A projection row's `detail` is a
 * label for the quantity, which the heading above it already names, and drawing
 * it twice is the defect rule 4 exists to end.
 *
 * **2. The detail adds a word the heading does not carry** —
 * `detailAddsToHeading`. Still load-bearing: `Sports_Predictor`'s two touchdown
 * categories are `kind: "probability"` with `detail` set to the heading's own
 * text, and without this half they print their heading three more times.
 *
 * **THE DEFECT THIS WAS FOUND WITH.** The first version of rule 4 was condition
 * 2 alone, and it passed the harness fixture — which used heading `"Rush yds —
 * projections, not probabilities"` with detail `"Rush yds"`, where the tokens DO
 * match. The real `Sports_Predictor` `origin/main` payload is
 * `{ category: "QB passing yards", detail: "Pass yds", kind: "projection" }`
 * (`src/lib/picksPanel.ts:79`, and the same shape at :80 and :81), so three of
 * that panel's categories drew an abbreviation of their own heading, AND drew
 * it with ` · model call` glued on, because the qualifier was applied to every
 * rendered detail regardless of kind. The fixture could not see either, because
 * no real row used an abbreviation. `src/components/sportsPayloads.test.tsx`
 * now holds the real rows and fails on both.
 *
 * **WHY NOT "tolerate abbreviations".** Considered and rejected, because every
 * version of it fails silently somewhere. A synonym table (`yds`→`yards`,
 * `rec`→`receiving`) is a sport's slang living in a component five sports share
 * — wrong for NBA and PL the day it meets them, and a new table for every
 * abbreviation any adapter invents. Prefix matching catches `pass`/`passing` and
 * `rec`/`receiving` but **not** `yds`/`yards`, so it would fix three of the
 * four words and leave the one that decides. And a digit heuristic ("a line has
 * a number in it") would suppress `3+ receptions`, which is information.
 * Condition 1 needs no table at all and is right about every one of them.
 *
 * **WHY NOT "callers pass a `detail` matching their own heading".** Correct, and
 * it is the cleanup worth doing — `positionCategories()` typing `detail` as the
 * category's own short label is the legacy meaning, and deleting it is the real
 * fix. Rejected as the fix here because it needs a `Sports_Predictor` PR, and
 * this package is vendored from `main`: between this merging and that landing,
 * three live categories would be wrong with nothing in the hub able to stop it.
 * Condition 1 is correct today, with no adapter change and no drift window.
 *
 * **WHAT IS GIVEN UP.** A probability row whose detail is built entirely from
 * words the heading already carries is dropped, so a caller cannot disambiguate
 * two rows in one list by such a word — `detail: "passing"` under `"QB passing
 * TDs"` draws nothing. Accepted: the heading is what the reader navigates by,
 * and `value` already separates two rows under one heading. And a projection row
 * can carry NO qualifier at all, even one its caller supplies, because the
 * qualifier is a claim that the row states a call.
 */
export function rowShowsDetail(row: PickRow, category: string): boolean {
  return row.kind === "probability" && detailAddsToHeading(row.detail, category);
}

/** A row marked `out` was passed as a ranked row: "removed, not flagged" is a
 *  rule about what reaches the page, and this is the last place that can say no. */
export class OutPlayerInRankingError extends Error {
  constructor(name: string) {
    super(
      `PicksList: "${name}" is marked out and was passed as a ranked row. An out player ` +
        `leaves the ranking entirely — pass them in \`out\` instead, where they appear once ` +
        `below the lists with their source and date.`,
    );
    this.name = "OutPlayerInRankingError";
  }
}

/** A row's `value` is outside the range its `kind` can honestly draw. */
export class RowKindMismatchError extends Error {
  constructor(player: string, detail: string, kind: "probability" | "projection", value: number) {
    super(
      kind === "probability"
        ? `PicksList: "${player}" is a ${detail} probability row, so its value must be in [0,1] — got ${value}. ` +
            `A yardage figure read as a share is the defect this guard exists for.`
        : `PicksList: "${player}" is a ${detail} projection row, so its value must be finite and not negative — got ${value}.`,
    );
    this.name = "RowKindMismatchError";
  }
}

/**
 * Refuse a row whose number cannot honestly draw the visual its `kind` asks for.
 *
 * Runs before anything renders, so a caller that made the mistake gets a named
 * error naming the player and the category instead of a bar reading 9600%.
 */
function checkRow(row: PickRow): void {
  if (row.out) throw new OutPlayerInRankingError(row.name);
  if (!Number.isFinite(row.value)) {
    throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  }
  if (row.kind === "probability") {
    if (row.value < 0 || row.value > 1) throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  } else {
    // A projection is a count or a distance: finite and not negative. It MAY be a
    // fraction (0.8 steals, 0.6 touchdowns). The row's `kind` is explicit, so a
    // magnitude test is not how a mislabelled row is caught here; refusing [0,1]
    // threw during render for legitimate rows and took the whole list down.
    if (row.value < 0) throw new RowKindMismatchError(row.name, row.detail, row.kind, row.value);
  }
}

/** The figure a row shows. A probability is a percentage; a projection is a plain
 *  number with no % anywhere, because a count read as a share is the "9600%" defect. */
function figure(row: PickRow): string {
  return row.kind === "probability" ? pct(row.value) : stat(row.value);
}

/** A thin bar for a probability only. Floored so a small probability stays visible. */
function Bar({ value }: { value: number }) {
  const width = Math.max(value * 100, 3);
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-pr-rule" aria-hidden="true">
      <div data-testid="picks-bar" className="h-full rounded-full bg-pr-accent" style={{ width: `${width}%` }} />
    </div>
  );
}

export function PicksList({ title = "Model's top calls", categories, out }: PicksListProps) {
  // Every row is checked before any of them renders, so one bad row cannot
  // leave a half-drawn list on the page.
  for (const { rows } of categories) for (const row of rows) checkRow(row);

  if (categories.length === 0 && !out?.length) return null;

  return (
    <div data-testid="picks-list" className="flex flex-col gap-4">
      <p
        data-testid="picks-title"
        className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint"
      >
        {title}
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        {categories.map(({ category, rows }) => {
          // The ceiling. `slice` rather than a filter, so an out player could not be
          // swapped in to backfill a dropped row; `checkRow` has already refused one.
          const shown = rows.slice(0, MAX_ROWS_PER_CATEGORY);
          return (
            <section
              key={category}
              data-testid="picks-category"
              className="flex flex-col gap-2 rounded-pr border border-pr-rule bg-pr-panel-2/40 p-3"
            >
              <h3
                data-testid="picks-category-heading"
                className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim"
              >
                {category}
              </h3>
              {shown.length === 0 ? (
                // The existing empty-state vocabulary, never a zero: a figure nobody
                // produced must not be drawn as one.
                <StatusBadge status="nopick" />
              ) : (
                shown.map((row, i) => (
                  <div
                    key={row.key}
                    data-testid="picks-row"
                    data-kind={row.kind}
                    data-category={category}
                    className="flex flex-col gap-1.5 border-t border-pr-rule pt-2 first:border-t-0 first:pt-0"
                  >
                    <div className="flex items-baseline gap-2">
                      <span data-testid="picks-rank" className="w-4 shrink-0 font-pr-display text-xs font-semibold text-pr-text-faint tabular-nums">
                        {i + 1}
                      </span>
                      <p className="min-w-0 flex-1 truncate text-sm font-medium leading-snug text-pr-text">
                        {row.name}
                        {row.team && <span className="font-normal text-pr-text-dim"> · {row.team}</span>}
                      </p>
                      <span
                        data-testid="picks-value"
                        className={`font-pr-display text-xl font-semibold leading-none tabular-nums ${i === 0 ? "text-pr-accent" : "text-pr-text"}`}
                      >
                        {figure(row)}
                      </span>
                    </div>
                    {/* Rule 4 — `rowShowsDetail`, and nothing else. A projection row
                        never reaches this branch, so it can never be labelled a
                        call, and `detailLabel` is never read for one. The label is
                        visible text rather than a `title`: see `detailLabel`. */}
                    {rowShowsDetail(row, category) && (
                      <p data-testid="picks-detail" className="pl-6 text-xs leading-snug text-pr-text-dim">
                        {row.detail}
                        <span className="text-pr-text-faint"> · {row.detailLabel ?? DEFAULT_DETAIL_LABEL}</span>
                      </p>
                    )}
                    {row.kind === "probability" && <Bar value={row.value} />}
                  </div>
                ))
              )}
            </section>
          );
        })}
      </div>

      {/* §D: an out player is shown ONCE, here, and nowhere in a ranking — no
          bar, no tile, no rank position, attributed and dated. */}
      {!!out?.length && (
        <div data-testid="picks-out" className="flex flex-col gap-1 border-t border-pr-rule pt-2">
          {out.map((p) => (
            <p key={`${p.name}-${p.dated}`} className="text-xs leading-snug text-pr-text-faint">
              Out: {p.name}
              {p.team && ` · ${p.team}`} — not ranked. {p.source} · {p.dated}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

/** The percentage a probability row shows, for a caller that wants the same
 *  rounding the bar uses. Exported so a site cannot re-derive it differently. */
export { pct as picksPct };