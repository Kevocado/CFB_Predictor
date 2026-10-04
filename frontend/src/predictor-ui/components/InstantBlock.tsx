// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** The facts, rendered before anything is asked for.
 *
 *  Every figure here comes from the site's own bundle or its `extras`. There
 *  is no request, no state and no effect: press the button or do not, this
 *  block is already true. That is the whole point of it — a reader should not
 *  have to spend a request to learn the probability the site already had.
 *
 *  Order is fixed: timing, verdict, tiles, bar, record. The timing row comes
 *  first because it governs how every number below it should be read, and the
 *  record comes last because it is about past picks, not this fixture.
 */
import { KeyNumberTile } from "./KeyNumberTile";
import { ProbabilityBar } from "./ProbabilityBar";
import { RecordStrip } from "./RecordStrip";
import { StatusBadge } from "./StatusBadge";
import { pickLabel, pickTiming, verdictSentence } from "../lib/bundleFacts";
import type { FixtureExtras } from "./FixtureExplainer";

export interface InstantBlockProps {
  sport: string;
  bundle: any;
  extras?: FixtureExtras;
}

const STARTED = {
  kickoff: "after kickoff",
  "tip-off": "after tip-off",
  "the session": "after the session started",
} as const;

export function InstantBlock({ sport, bundle, extras }: InstantBlockProps) {
  const timing = pickTiming(bundle, sport);
  const verdict = verdictSentence(bundle, sport);
  const tiles = extras?.tiles ?? [];
  const segments = extras?.segments;
  // A record alone is still something to show: a bundle with no pick can carry
  // the record strip, and a guard that ignored it dropped the record silently.
  if (!verdict && tiles.length === 0 && !segments && !extras?.record) return null;
  // The pick the BUNDLE names, never worked out here: the bar accents the segment
  // whose label matches it, and nothing when there is no pick.
  const pick = pickLabel(bundle, sport);

  return (
    <div className="flex flex-col gap-3" data-testid="instant-block">
      {timing && timing.status === "none" && (
        <p className="flex items-center gap-2 text-sm text-pr-text-dim">
          <span className="inline-flex items-center whitespace-nowrap rounded-pr border border-pr-rule px-1.5 py-0.5 font-pr-display text-xs font-semibold uppercase tracking-wide">
            Made before {timing.moment === "the session" ? "the session" : timing.moment === "tip-off" ? "tip-off" : "kickoff"}
          </span>
        </p>
      )}
      {timing?.status === "rebuilt" && (
        <p className="flex flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="rebuilt" moment={timing.moment} />
          {/* The moment, and the fact that it counts. This line used to say the
            pick was shown only as a reference and held out of the count, which
            is the opposite of the rule: the track record counts the earliest
            recorded pick per (game, market) whenever it was made (spec
            2026-10-01-track-record-counts-every-pick). The badge beside this
            line already carries the moment, so what the sentence adds is that
            the pick is not being held back from the record — the two halves of
            disclosure, which is what replaced exclusion. The old words are
            spelled out in `countedCopy.test.tsx`, which guards against them
            coming back. */}
          <span>
            This pick was made {STARTED[timing.moment]}. Counted in the track record like any other pick.
          </span>
        </p>
      )}
      {timing?.status === "unverified" && (
        <p className="flex flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="unverified" moment={timing.moment} />
          <span>The schedule did not provide the start time, so this pick cannot be shown as made before the start.</span>
        </p>
      )}
      {verdict && <p className="max-w-[70ch] text-lg font-medium leading-snug">{verdict}</p>}
      {timing?.status === "nopick" && (
        <p className="text-sm text-pr-text-dim">No pick was made for this fixture.</p>
      )}
      {tiles.length > 0 && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {tiles.map((t) => <KeyNumberTile key={t.market} tile={t} />)}
        </div>
      )}
      {segments && segments.length > 0 && (
        <ProbabilityBar segments={segments} legend={extras?.legend} minSegmentPx={2} pick={pick ? { label: pick } : null} />
      )}
      {extras?.record && (
        <RecordStrip label={extras.record.label} hits={extras.record.hits} settled={extras.record.settled} />
      )}
    </div>
  );
}

export const AI_PROMISE = "AI read: model vs line, trends, who's out.";
