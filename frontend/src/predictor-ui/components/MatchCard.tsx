// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";
import { pct } from "../fmt";
import { ProbabilityBar, type Segment } from "./ProbabilityBar";
import { StatusBadge, type Moment, type Status } from "./StatusBadge";
import { TeamChip } from "./TeamChip";

export type Side = { code: string; name?: string; color?: string; badge?: ReactNode };

type Props = {
  left: Side;
  right: Side;
  /** Kickoff time before the game, the score after it. */
  centre: string;
  /** Omit for an upcoming game that is not next up: no badge. */
  status?: Status;
  pick?: { label: string; prob: number };
  /** Top-left line, usually the kickoff date. */
  when?: string;
  /** Sport-specific line under the pick: a spread, a total, a venue. */
  meta?: string;
  bar?: Segment[];
  /** Shown instead of a pick when there is none yet (default "No pick yet"). */
  pickPlaceholder?: string;
  /** Status wording: "Made after kickoff" or "… tip-off". */
  moment?: Moment;
  /** Tighter card for nights with many games (NBA). Same content. */
  compact?: boolean;
  onOpen: () => void;
};

// The card is named by its own visible text (WCAG 2.5.3 Label in Name), so
// screen readers hear everything sighted visitors see. Buttons may only hold
// phrasing content, so the layout is built from spans.
const Sep = () => <span className="sr-only">, </span>;

function Team({ side }: { side: Side }) {
  return (
    <span className="flex min-w-0 flex-1 flex-col items-center gap-1.5 text-center">
      <span aria-hidden="true">{side.badge ?? <TeamChip code={side.code} color={side.color} />}</span>
      <span className="block w-full break-words text-sm font-semibold leading-tight text-pr-text [text-wrap:balance]">{side.name ?? side.code}</span>
    </span>
  );
}

/** The family match card: one pick, one status, a labelled bar. */
export function MatchCard({ left, right, centre, status, pick, when, meta, bar, pickPlaceholder = "No pick yet", moment = "kickoff", compact = false, onOpen }: Props) {
  // A missing pick is always said out loud; the status badge only covers it
  // when the badge itself reads "No pick yet".
  const pickLine = pick ? `Pick: ${pick.label} · ${pct(pick.prob)}` : status === "nopick" ? undefined : pickPlaceholder;

  return (
    <button
      type="button"
      onClick={onOpen}
      data-compact={compact || undefined}
      className={`pr-notch flex w-full flex-col rounded-pr border border-pr-rule bg-pr-panel text-left transition-colors hover:border-pr-accent ${
        compact ? "gap-2 p-3" : "gap-3 p-4"
      }`}
    >
      <span className="flex w-full flex-wrap items-center justify-between gap-x-2 gap-y-1">
        <span className="text-xs uppercase tracking-wide text-pr-text-dim">{when}</span>
        {status && (
          <>
            <Sep />
            <StatusBadge status={status} moment={moment} />
          </>
        )}
      </span>
      <Sep />
      <span className="flex w-full items-center gap-2">
        <Team side={left} />
        <Sep />
        <span
          className={`shrink-0 rounded-pr bg-pr-panel-2 px-2.5 py-1 font-pr-display font-semibold tracking-wide text-pr-text ${
            compact ? "text-lg" : "text-xl"
          }`}
        >
          {centre}
        </span>
        <Sep />
        <Team side={right} />
      </span>
      {(pickLine || bar || meta) && (
        <span data-testid="pick-section" className="flex w-full flex-col gap-2 border-t border-pr-rule pt-2.5">
          <Sep />
          {pickLine && <span className={`text-sm font-semibold ${pick ? "text-pr-text" : "text-pr-text-dim"}`}>{pickLine}</span>}
          {/* No `onSegmentFocus`, which is what keeps the bar's segment labels as
              text rather than buttons: a button inside this card's own `<button>`
              is invalid HTML, and a browser is entitled not to make the inner one
              interactive at all. It would also be a control with nothing to
              report to — §13c's other half is a factor list, and a card has no
              factors, so there is no highlight for a focused segment to join and
              nothing for it to dim. The card's own surface is the affordance here,
              and the figures stay visible either way. Passing a handler on this
              line is what puts the nesting back. */}
          {bar && <ProbabilityBar segments={bar} pick={pick ? { label: pick.label } : null} />}
          {meta && (
            <>
              <Sep />
              <span className="text-xs text-pr-text-dim">{meta}</span>
            </>
          )}
        </span>
      )}
    </button>
  );
}
