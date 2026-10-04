// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState } from "react";
import { contrast, parseHex } from "../contrast";
import { pct } from "../fmt";

export type Segment = {
  label: string;
  prob: number;
  color?: string;
  /** The facts' `market` key, when this segment IS a market. A factor's `key`
   *  refers to it, which is what makes §13c's highlight a lookup rather than a
   *  second naming scheme. Optional, so the match-card call sites that predate
   *  the panel are unaffected. */
  market?: string;
};

// Which outcome the model picked, as the service derived it from the facts
// (spec §5b; `contract.pick_for`). The panel never works this out for itself:
// a renderer that guessed the pick from segment order or from the widest
// segment is the defect this field exists to remove.
// Not named `Pick`, which would shadow the built-in utility type of that name
// inside every file that imports it. The key on the wire is still `pick`.
export type PickRef = { label: string; side?: string };

/** The only colour that means "this is the pick", so only the pick may spend
 *  it — and a bar with no pick spends none of it. */
const ACCENT = "var(--color-pr-accent)";

// The neutral ramp a segment that is NOT the pick is painted in, most prominent
// first. Never good/bad green/red: a green segment beside a grey one reads as
// "right" and a red one as "wrong", which is a claim about the outcome, and
// this bar is only claiming which side was picked.
// Three steps, not two, because a three-way market (PL home/draw/away) has to
// keep three distinguishable tones once the accent has been given to the pick.
// Two segments painted the same grey is a worse defect than the one this
// replaces: the reader cannot read a 48% home off a 26% draw by tone at all.
// `fill-mute` is the third rung, added to tokens.css for exactly this.
const NEUTRALS = [
  "var(--color-pr-text-dim)",
  "var(--color-pr-text-faint)",
  "var(--color-pr-fill-mute)",
];

// A team colour that nearly matches the panel (Browns brown, Ravens purple)
// would make its share of the bar disappear; those fall back to a neutral
// that reads. 1.6:1 against the panel is the floor.
// Kept equal to --color-pr-panel by tokens.test.ts.
export const PANEL = "#151920";
function fill(color: string | undefined, emphasised: boolean, rank: number): string {
  const hex = color ? parseHex(color) : null;
  return hex && contrast(hex, PANEL) >= 1.6 ? hex : emphasised ? ACCENT : NEUTRALS[rank % NEUTRALS.length];
}

/** Where the pick is, as an index into `segments`; -1 when there is no pick to
 *  find, and -1 for a pick whose label matches nothing here too. That second
 *  case fails closed on purpose: a label the bar cannot find is a site passing
 *  a different vocabulary, and pointing at the nearest segment to it would put
 *  the emphasis on a claim nobody made. */
function pickIndex(segments: Segment[], pick: PickRef | null | undefined): number {
  return pick ? segments.findIndex((s) => s.label === pick.label) : -1;
}

/** One tone per segment, in the order given.
 *
 *  The ramp is walked by the segments that are NOT the pick, so the neutrals
 *  stay three distinct tones on a three-way bar whether the pick is the first
 *  segment, the last, or the draw in the middle. A segment with a legible team
 *  colour is not painted from the ramp at all and does not consume a step, so
 *  the neutrals beside it still start at the most prominent rung.
 *
 *  A bar of five outcomes or more would wrap the ramp and repeat a tone. Nothing
 *  this product serves draws more than three outcomes on one market, and a
 *  repeat is what the old index-based fallback did too — it is not a new
 *  collapse, so it is left rather than guarded.
 */
function tones(segments: Segment[], index: number): string[] {
  let rank = 0;
  return segments.map((s, i) => {
    const emphasised = i === index;
    const tone = fill(s.color, emphasised, rank);
    if (!emphasised) rank += 1;
    return tone;
  });
}

/** Whether the market's own split covers exactly the outcomes the model's does.
 *
 *  Spec §13b. PL's `implied` carries only the two sides, so with a three-way
 *  model split a market row would sit under a draw segment with nothing above
 *  it: two bars covering different outcomes, inviting a comparison that cannot
 *  be made. Omission is the honest answer.
 *
 *  **Both directions, not one.** §13b says the market row covers every outcome
 *  the model's split has, and a *superset* satisfies that sentence while
 *  breaking the thing the row is for: a market segment with no column above it
 *  cannot be compared with anything, and it also makes the figures below the two
 *  bars stop lining up. So the two splits must be the same set of outcomes, one
 *  each, and the figures are then read by label rather than by position.
 */
function covers(segments: Segment[], legend: Segment[]): boolean {
  const have = new Set(legend.map((s) => s.label));
  return segments.length > 0 && legend.length === segments.length && segments.every((s) => have.has(s.label));
}

/** The market's own figures, in the model's outcome order, so each one sits
 *  under the model's figure it belongs to.
 *
 *  **Read by label, never by position.** `segments` and `legend` are two
 *  different facts, and a call site that hands them over in a different order is
 *  not a mistake to paper over — it is precisely the case where pairing them by
 *  index would print the market's 44% under the model's 48% and still look like
 *  a readable comparison. `covers` has already established that every label
 *  matches, so the lookup cannot miss. */
function marketSplit(segments: Segment[], legend: Segment[]): Segment[] {
  return segments.map((s) => legend.find((l) => l.label === s.label)!);
}

/** What a segment's figures wear, in both branches below. One constant, because
 *  the only thing that differs between a label that is a control and a label
 *  that is text is the element and its handlers — not one pixel of it. */
const LABEL_CLASS =
  "min-w-0 rounded-pr text-xs tabular-nums text-pr-text-dim underline-offset-4 transition-opacity duration-150 hover:text-pr-text focus-visible:text-pr-text";

/** One segment's figures: a control when somebody is listening for it, and text
 *  when nobody is.
 *
 *  **Why the buttons are conditional, and why the condition is `onSegmentFocus`
 *  rather than a prop.** Those buttons have exactly one job — §13c's step
 *  through the figures, which reports a focus up to `onSegmentFocus` and gets a
 *  `highlightKey` highlight back. A `<button>` whose only handler is
 *  `onSegmentFocus?.()` with nothing passed is a control that is announced,
 *  reachable, and does nothing: a focus stop that lies. `MatchCard` was exactly
 *  that, and it was not a near-miss: a card cannot host §13c at all, because the
 *  other half of §13c is a `FactorList` and a card has no factors. So the card
 *  never passed `onSegmentFocus`, never will, and the buttons inside it were
 *  dead controls inside a `<button>` — invalid HTML a browser is entitled to
 *  stop making interactive, which is the same as saying a keyboard reader has
 *  stops that do nothing.
 *
 *  A prop would be a thing every call site has to remember, and here the default
 *  is what a call site that remembers nothing gets. Deriving the branch from a
 *  signal the component already receives makes the unsafe answer unreachable *for
 *  a caller that passes no listener*: a bar in a button with nothing to report gets
 *  plain text — valid HTML, no dead focus stop. A prop defaulting to interactive
 *  would fail in the other direction, into invalid HTML, on the call site that
 *  forgot.
 *
 *  **What it does not reach, and the one prop that does.** "Unreachable by
 *  accident" was this comment's claim, and it was too wide: a component cannot see
 *  its own ancestors, so nothing derived from `onSegmentFocus` can protect a
 *  caller who *does* pass a listener and renders the bar inside a control. That
 *  combination is `button > button` — measured, in `nesting.test.tsx`, with this
 *  very component — and `MatchCard` is protected from it only because a card has
 *  no factors and so never has a listener. `insideControl` is the one thing here a
 *  caller has to remember, and it is a real cost taken deliberately: the
 *  alternative is not "the default is safe", it is "the default is safe only for
 *  the call sites that happen to have nothing to report". There is no CSS answer
 *  either — a `<button>` is a `<button>` however its contents are styled.
 *
 *  **Nothing is withheld either way.** The figures are in this label and in the
 *  bar's accessible name above, so a reader who cannot step through them still
 *  has every one of them. What a non-interactive label gives up is a way to
 *  *arrive* at them one at a time, which is a duplicate of the affordance
 *  whatever surface the bar is on. The market row's disclosure is the same: it
 *  cannot exist inside a control at all, so the figures there stay open rather
 *  than being hidden behind a button the caller is not allowed to render. */
function SegmentFigure({
  segment,
  highlighted,
  opacity,
  onFocus,
}: {
  segment: Segment;
  highlighted: boolean;
  opacity: number;
  onFocus?: (label: string, market?: string) => void;
}) {
  const figures = (
    <>
      {segment.label} {pct(segment.prob)}
    </>
  );
  if (!onFocus) {
    return (
      <span
        data-testid="pbar-label"
        data-seg={segment.label}
        data-highlighted={highlighted ? "true" : "false"}
        className={LABEL_CLASS}
        style={{ opacity }}
      >
        {figures}
      </span>
    );
  }
  return (
    <button
      type="button"
      data-testid="pbar-label"
      data-seg={segment.label}
      data-highlighted={highlighted ? "true" : "false"}
      onClick={() => onFocus(segment.label, segment.market)}
      onFocus={() => onFocus(segment.label, segment.market)}
      className={LABEL_CLASS}
      style={{ opacity }}
    >
      {figures}
    </button>
  );
}

/**
 * Segments render in the order given, each labelled in words.
 *
 * The graphic keeps its `role="img"` and its single label, which is the right
 * summary for a reader who is not going to step through it. The per-segment
 * labels are then operable — real buttons carrying the same figures, so a
 * reader who *does* step through it gets them one at a time — whenever there is
 * a caller to report that step to; see `SegmentFigure` for why that is the
 * condition. That is summary plus detail, and it is why nothing here is
 * `aria-hidden`: a focusable element inside an `aria-hidden` container is
 * reachable by keyboard and invisible to a screen reader, which is worse than
 * either alone.
 */
export function ProbabilityBar({
  segments,
  legend,
  minSegmentPx = 0,
  highlightKey = null,
  pick = null,
  expandable = false,
  onSegmentFocus,
  insideControl = false,
}: {
  segments: Segment[];
  legend?: Segment[];
  /** A 3% probability still has to render as a sliver rather than vanishing.
   *  §6a: the label goes in the legend rather than shrinking below the 12px
   *  floor, so visibility and the floor are not in tension. */
  minSegmentPx?: number;
  /** The market key a selected factor points at (§13c). */
  highlightKey?: string | null;
  /** The pick, from the answer. The accent follows it and nothing else does:
   *  segment order used to decide, so on a two-way bar with the away side
   *  listed first the model was shown to have picked the home side. Absent
   *  means no pick, and then no segment is accented — a bar that emphasises
   *  something is claiming there is a pick, and with none there is none to
   *  point at. */
  pick?: PickRef | null;
  /** Narrow layouts can collapse the market row's figures, the same way
   *  `FactorList` clamps a factor's sentence: a real `<button aria-expanded>`,
   *  off by default so a wide layout never hides text behind a control that adds
   *  nothing. Open to begin with, because an unlabelled market bar is the defect
   *  the figures exist to fix.
   *
   *  **It stays opt-in, and a CSS-only replacement was considered and rejected.**
   *  The hazard is real and it is the one this component does not design out: a
   *  site that forgets this prop gets a `justify-between` row that collides at
   *  260px instead of the fallback, and the guard is remembering a prop rather
   *  than like `onSegmentFocus` (above) being derived from a signal already
   *  present. Three CSS answers were measured against the spec and each fails it:
   *
   *  - **`clamp()`** — ruled out by §6a, which sets a 12px floor and says the
   *    market row "does not get a smaller size to make a row fit, it drops the
   *    row". Both consuming sites enforce the floor in their own tests
   *    (`craft-floor.test.ts` in Sports_Predictor, `hub.test.mjs` in the Hub), so
   *    this is not a style preference with no teeth.
   *  - **A container query that hides the figures.** It fixes the collision, but
   *    CSS cannot add the control that brings them back, so on a narrow surface it
   *    withholds data with no way to reach it — the failure §13c's closing
   *    constraint is written about. The disclosure has to be a real element, which
   *    is the whole reason the fallback is a button and not a `hidden` attribute.
   *  - **A container query that reflows instead** (`flex-wrap`, or the cells
   *    stacking). Nothing is hidden, so nothing is withheld, and the 12px floor
   *    holds. But it breaks the one thing the row exists for: spec §6 item 3 says
   *    each figure sits "in the column of the figure it is being compared with",
   *    and a wrapped row puts two outcomes on one line and one on the next, so the
   *    comparison stops lining up. At 260px with three outcomes there is no layout
   *    that keeps three 12px figures in three columns; something has to give, and
   *    §13c item 4 already chose what.
   *
   *  So the cost is a prop to remember, and the compensation is that the default
   *  is the *safe* one: off means the figures are shown, which is a row that reads
   *  well and a site that has not yet discovered it is narrow. The failure mode of
   *  forgetting is a collision at 260px — a layout defect a screenshot catches —
   *  and not withheld text, a dead control, or a claim. The rule stays where §13c
   *  item 4 put it, and this is the note so the next person does not re-open it
   *  without measuring first. */
  expandable?: boolean;
  /** §13c's step through the figures. **This is also what makes the segment
   *  labels operable**: with it, each label is a real button that reports its
   *  own focus here, and `highlightKey` lights the figure a factor named. With
   *  nothing passed, the labels are text — a button with no listener is a focus
   *  stop that announces a control and does nothing, and on a surface that is
   *  already a button (a match card) it is also invalid HTML. */
  onSegmentFocus?: (label: string, market?: string) => void;
  /** This bar is rendered **inside another control** — a `<button>`, an `<a
   *  href>`, a `role="button"` — so it must render no controls of its own: every
   *  label becomes text and the market row's disclosure does not exist.
   *
   *  There is no way for a component to detect this, so the caller has to say it.
   *  A bar with no `onSegmentFocus` inside a button is already safe and does not
   *  need this; a bar *with* one is not, and a `<button>` inside a `<button>` is
   *  invalid HTML a browser is entitled to stop making interactive — which is the
   *  same as saying a keyboard reader has stops that do nothing. `nesting.test.tsx`
   *  pins both halves: this case is green with the prop, and is a fault without
   *  it. */
  insideControl?: boolean;
}) {
  // Open unless the caller says the layout is narrow. Held here rather than
  // derived from a width, because `FactorList` made the same call: a CSS-only
  // clamp is not announced, and a measurement the panel never takes is a
  // measurement that can be wrong in the reader's favour.
  // Inside a control there is no disclosure to be open or closed — the button
  // that would close it is the thing this bar is not allowed to render — so the
  // figures stay open rather than being hidden with no way back to them.
  const [figuresOpen, setFiguresOpen] = useState(true);
  const index = pickIndex(segments, pick);
  const fills = tones(segments, index);
  const named = segments.map((s) => `${s.label} ${pct(s.prob)}`);
  // The accent is a colour, so the emphasis has to be said as well. This is the
  // same rule FactorList states about its triangles: a meaning carried by
  // colour alone is not carried at all for a reader who cannot see it, in
  // greyscale print, or on the wrong background. Only when the pick actually
  // matched a segment — an unmatched label is not a claim about any of them.
  const described = index < 0 ? named : [...named, `the pick is ${segments[index].label}`];
  const showLegend = !!legend && covers(segments, legend);
  // The market's figures, from `legend` and never from `segments`. The two bars
  // are a comparison, so the row underneath quotes the *other* split: a figure
  // read off the wrong array would be the right number attributed to the wrong
  // source, which is the one failure on this row that is worse than no row.
  const market = showLegend ? marketSplit(segments, legend!) : [];
  const showToggle = expandable && !insideControl;
  const showFigures = showLegend && (!showToggle || figuresOpen);
  // Operable only where a control is legal. Inside another control the labels are
  // text, and they keep the figures and the highlight — everything except the
  // ability to *arrive* at them one at a time, which the surface wrapping them
  // already provides.
  const focus = insideControl ? undefined : onSegmentFocus;
  const dim = (s: Segment) => (highlightKey && s.market && s.market !== highlightKey ? 0.4 : 1);

  return (
    <div className="flex flex-col gap-1.5">
      <div role="img" aria-label={described.join(", ")}>
        <span aria-hidden="true" className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-pr">
          {segments.map((s, i) => (
            <span
              key={i}
              data-testid="pbar-fill"
              data-min-width={minSegmentPx ? `${minSegmentPx}px` : undefined}
              className="h-full transition-opacity duration-150"
              style={{
                flexGrow: Math.max(s.prob, 0.01),
                minWidth: minSegmentPx ? `${minSegmentPx}px` : undefined,
                backgroundColor: fills[i],
                opacity: dim(s),
              }}
            />
          ))}
        </span>
      </div>

      <div className="flex items-baseline justify-between gap-2">
        {segments.map((s, i) => (
          <SegmentFigure
            key={i}
            segment={s}
            highlighted={!!highlightKey && s.market === highlightKey}
            opacity={dim(s)}
            onFocus={focus}
          />
        ))}
      </div>

      {showLegend && (
        <div data-testid="pbar-legend" className="flex flex-col gap-1 border-t border-pr-rule pt-1.5">
          {/* The word names the row; the bar beneath it is then the full width of
              the bar above, so the two splits start and end in the same place
              and a reader can see the difference without measuring. A gutter
              beside the word would indent this bar out of comparison with the
              model's, which is the one thing the row exists to make possible. */}
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-xs uppercase tracking-wide text-pr-text-faint">market</span>
            {showToggle && (
              <button
                type="button"
                data-testid="pbar-market-toggle"
                aria-expanded={showFigures}
                onClick={() => setFiguresOpen((was) => !was)}
                className="text-xs font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 hover:underline"
              >
                {showFigures ? "Less" : "More"}
              </button>
            )}
          </div>
          {/* The market bar gets no `minSegmentPx`, and that is deliberate. A
              floor on the *comparison* bar would draw a 1% implied share as wide
              as the model's 3% sliver, which is the one thing this row must not
              do: the labels below carry the figures, and the geometry carries
              only the shape. */}
          <div role="img" aria-label={`market: ${market.map((s) => `${s.label} ${pct(s.prob)}`).join(", ")}`}>
            <span aria-hidden="true" className="flex h-1 w-full gap-0.5 overflow-hidden rounded-pr">
              {market.map((s, i) => (
                <span
                  key={i}
                  data-testid="pbar-market-fill"
                  style={{ flexGrow: Math.max(s.prob, 0.01), backgroundColor: "var(--color-pr-text-faint)" }}
                />
              ))}
            </span>
          </div>
          {/* The figures, in the same row shape and the same tone as the labels
              above so each one sits in the column of the figure it is being
              compared with. `text-xs` is the 12px floor, here exactly as it is
              on the model's own labels: §6a rules out shrinking a figure that
              will not fit, and the way to survive a narrow row is to drop the
              row behind the toggle rather than to set it smaller. */}
          {showFigures && (
            <div data-testid="pbar-market-figures" className="flex items-baseline justify-between gap-2">
              {market.map((s) => (
                <span
                  key={s.label}
                  data-market={s.label}
                  className="min-w-0 text-xs tabular-nums text-pr-text-faint"
                >
                  {s.label} {pct(s.prob)}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
