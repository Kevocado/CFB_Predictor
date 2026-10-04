// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
export type Status = "next" | "live" | "called" | "missed" | "nopick" | "rebuilt" | "unverified";
/** The moment a pick must beat: "kickoff" for football, "tip-off" for
 *  basketball, "the session" for F1 (qualifying and races alike). */
export type Moment = "kickoff" | "tip-off" | "the session";

// Status always carries words; colour only reinforces them.
const LOOK: Record<Status, { words: string; tone: string }> = {
  next: { words: "Next up", tone: "bg-pr-accent text-pr-accent-ink" },
  live: { words: "Live", tone: "border border-pr-loss text-pr-loss" },
  called: { words: "Called it ✓", tone: "text-pr-win" },
  missed: { words: "Missed ✗", tone: "text-pr-loss" },
  nopick: { words: "No pick yet", tone: "text-pr-text-dim" },
  // The KEY is still `rebuilt` and is not copy: it is the value a site passes to
  // `<StatusBadge status=…>` and the value `pick_timing` carries on the wire, so
  // renaming it would break every site's compile to change nothing a reader
  // sees. The WORDS are what changed. Since the track record began counting the
  // earliest recorded pick per (game, market) whatever moment it was made
  // (spec 2026-10-01-track-record-counts-every-pick), the honest description of
  // this state is when the pick was made, not that it was rebuilt afterwards.
  rebuilt: { words: "Made after {moment}", tone: "border border-pr-rule text-pr-text-dim" },
  // An unverified pick is not a fault: the schedule never supplied the start
  // time, so nothing went wrong. Neutral grey like `rebuilt`, not red — a red
  // badge would invent a failure that isn't there.
  unverified: { words: "Timing not confirmed", tone: "border border-pr-rule text-pr-text-dim" },
};

/** The exact words each status reads as, for accessible names elsewhere. */
export const statusWords = (status: Status, moment: Moment = "kickoff") => LOOK[status].words.replace("{moment}", moment);

export function StatusBadge({ status, moment = "kickoff" }: { status: Status; moment?: Moment }) {
  const { tone } = LOOK[status];
  const words = statusWords(status, moment);
  return (
    <span className={`inline-flex items-center whitespace-nowrap rounded-pr px-1.5 py-0.5 font-pr-display text-xs font-semibold uppercase tracking-wide ${tone}`}>
      {words}
    </span>
  );
}
