// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useState } from "react";

/**
 * The "why it matters" rows: one per factor, each naming the market it is about.
 *
 * Three things this component is careful about, and each is a rule rather than a
 * preference:
 *
 * 1. **The direction is never carried by colour.** A drawn triangle and a word —
 *    "for the pick" / "against it" — so the meaning survives with no colour
 *    perception, no greyscale print and no stylesheet. The triangle is
 *    `aria-hidden` and the words are the accessible truth. A `neutral` factor
 *    draws neither: there is no direction to draw, and a mark in either
 *    direction would be one.
 * 2. **The marks are drawn, not borrowed.** The §7a mock uses ▲ and ▼, which are
 *    unicode characters standing in for an icon system: they render at the
 *    viewer's font, at the viewer's weight, and they are not a consistent set.
 *    These are two authored triangles on one 24-unit grid, so up and down are
 *    visibly the same mark turned over.
 * 3. **A row is a real button.** §13c links a factor to the figure it names, and
 *    the second half of every affordance in this panel is that it is operable by
 *    keyboard alone. That assertion is what catches a `div` with an `onClick`.
 */
export type Factor = {
  key: string;
  /** Relative to the pick, per spec §5a: `up` argues for it, `down` against,
   *  and `neutral` neither — a statement about the game (the total, both teams
   *  to score) or about the record. With no pick every factor is `neutral`,
   *  because there is nothing to be for or against. */
  direction: "up" | "down" | "neutral";
  headline: string;
  text: string;
};

const TRIANGLE = (up: boolean) => (
  <svg
    data-direction={up ? "up" : "down"}
    aria-hidden="true"
    viewBox="0 0 24 24"
    width="14"
    height="14"
    className="shrink-0 translate-y-px"
    fill="currentColor"
  >
    <path d={up ? "M12 6.5 21 19H3z" : "M12 17.5 3 5h18z"} />
  </svg>
);

/** The mark a direction is allowed, keyed by direction rather than derived from
 *  it. `direction === "up"` used to decide, which reads a neutral row as "not
 *  up" and so paints it as a down triangle in the loss colour — turning a row
 *  that claims nothing into a claim that the case is against the pick. A table
 *  keyed by the whole union cannot do that: there is no default branch to fall
 *  into, so a fourth direction fails to compile rather than to render as a
 *  verdict. `neutral` has no mark at all. */
const MARK: Record<Factor["direction"], "up" | "down" | null> = {
  up: "up",
  down: "down",
  neutral: null,
};

/** The colour a mark is allowed. Green means "for" and red means "against", so
 *  a neutral row wears neither: the panel's own faint text token, which is the
 *  same one its direction words are already set in. */
const TONE: Record<Factor["direction"], string> = {
  up: "text-pr-win",
  down: "text-pr-loss",
  neutral: "text-pr-text-faint",
};

const WORDS: Record<Factor["direction"], string> = {
  up: "for the pick",
  down: "against it",
  // A label, not a direction, because that is what the row now is: a statement
  // about the game, the record, or how little there is to weigh up. It must not
  // name a pick — with no pick there is none to name — and it must not read as a
  // verdict. Rejected: "neutral", which in this product's copy means "no edge",
  // i.e. a claim about the pick; "not for or against", which is accurate and
  // re-names the pick in exactly the state that has none; "either way", which
  // claims the row cancels itself out when it does not; and "for reference",
  // which reads as a hedge on whether the row is worth anything — the very
  // framing the track record change removed, where a pick made after the start
  // was held out of the record and hedged with that phrase (spec
  // 2026-10-01-track-record-counts-every-pick). Two different warnings would
  // have blurred into one word.
  neutral: "context",
};

export function FactorList({
  factors,
  onSelect,
  highlighted,
  expanded,
  onToggle,
  expandable = false,
}: {
  factors: Factor[];
  onSelect?: (key: string) => void;
  /** The `key` of the factor whose figures the panel has lit, or null. Owned by
   *  the panel — it is the other end of §13c's linkage, the same `highlighted`
   *  value the tile and the bar segment are given.
   *
   *  **This is what the row's pressed state means.** It used to track `expanded`,
   *  which is a different state that the reader had not touched on a wide panel,
   *  so the row someone had just clicked gave no sign that the panel had changed.
   *  Expansion still has its own control with `aria-expanded` below, so nothing is
   *  lost by taking `data-highlighted` and `aria-pressed` for the selection: a
   *  button's pressed state should say what pressing it did, and what pressing a
   *  *why* row does is light a figure. */
  highlighted?: string | null;
  /** Which factor's sentence is fully shown. Owned by the panel, so a factor can
   *  be highlighted and expanded independently. */
  expanded?: number | null;
  onToggle?: (index: number) => void;
  /** Narrow viewports clamp the sentence to its first line. Off by default, so a
   *  wide layout never hides text behind a control that adds nothing. */
  expandable?: boolean;
}) {
  const [own, setOwn] = useState<number | null>(null);
  const isOpen = (i: number) => (expanded !== undefined ? expanded === i : own === i);
  const toggle = (i: number) => {
    if (onToggle) onToggle(i);
    else setOwn((was) => (was === i ? null : i));
  };

  return (
    <ul className="flex flex-col gap-3">
      {factors.map((factor, i) => {
        const mark = MARK[factor.direction];
        const open = isOpen(i);
        // The row's own half of §13c: this row is the one that asked for the
        // light, so it is the one that is pressed. Left out, pressing a *why* row
        // changes something two hundred pixels away and nothing here, and the
        // reader is left to infer from the figures that they did anything at all.
        const lit = !!highlighted && highlighted === factor.key;
        return (
          <li key={`${factor.key}-${i}`} className="flex min-w-0 flex-col gap-1">
            <div className="flex items-start gap-2">
              {/* The gutter stays whatever the mark is, so a neutral row's words
                  line up with the rows above and below it. */}
              <span className={`mt-0.5 w-[14px] shrink-0 ${TONE[factor.direction]}`}>
                {mark && TRIANGLE(mark === "up")}
              </span>
              {onSelect ? (
                <button
                  type="button"
                  data-testid={`factor-${factor.key}`}
                  data-highlighted={lit ? "true" : "false"}
                  onClick={() => {
                    onSelect(factor.key);
                    if (expandable) toggle(i);
                  }}
                  aria-pressed={lit}
                  className="min-w-0 flex-1 text-left text-sm font-semibold text-pr-text underline-offset-4 hover:underline"
                >
                  <span>{factor.headline}</span>{" "}
                  {/* The words carry the direction, so the meaning is never
                      carried by the colour of the triangle beside it. */}
                  <span className="text-xs font-normal uppercase tracking-wide text-pr-text-faint">
                    {WORDS[factor.direction]}
                  </span>
                </button>
              ) : (
                <h4 className="min-w-0 flex-1 text-sm font-semibold text-pr-text">
                  {factor.headline}{" "}
                  <span className="text-xs font-normal uppercase tracking-wide text-pr-text-faint">
                    {WORDS[factor.direction]}
                  </span>
                </h4>
              )}
            </div>
            <div className="flex items-start gap-2">
              <span aria-hidden="true" className="w-[14px] shrink-0" />
              <div className="min-w-0 flex-1">
                <p
                  data-expanded={open ? "true" : "false"}
                  className={`text-sm leading-relaxed text-pr-text-dim ${
                    expandable && !open ? "line-clamp-1" : ""
                  }`}
                >
                  {factor.text}
                </p>
                {expandable && (
                  <button
                    type="button"
                    onClick={() => toggle(i)}
                    aria-expanded={open}
                    className="mt-0.5 text-xs font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 hover:underline"
                  >
                    {open ? "Less" : "More"}
                  </button>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
