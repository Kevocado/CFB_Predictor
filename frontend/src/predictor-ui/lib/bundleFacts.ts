// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** Parsing of the flow bundle that both the instant block and the flow need.
 *
 *  The bundle shape differs per site (PL names `team_home`/`pick` as a string,
 *  Sports names `home_team`/`pick` as an object, F1 names a `driver`). These
 *  four helpers are the ONE place that knows those differences, so the block
 *  and the flow cannot disagree about what the bundle says — the failure mode
 *  that produced 68 on one part of a page and 72 on another.
 *
 *  Nothing here derives a probability: a bundle without a pick probability
 *  yields null, and the caller renders nothing rather than a number nobody
 *  stored.
 */
import type { Moment } from "../components/StatusBadge";

/** The four readings the block can state about a pick's timing. Deliberately
 *  narrower than `Status`: "none" here means *nothing to disclose* — the pick
 *  was made before the start — which is not the panel's `next`/`live` badge
 *  vocabulary and must not borrow it. */
export type Timing = { status: "rebuilt" | "unverified" | "nopick" | "none"; moment: Moment };

const MOMENT_OF: Record<string, Moment> = { f1: "the session", nba: "tip-off" };

/** The pick as the bundle words it — the abbreviation a site's segments are
 *  labelled in, and the string the flow joins against. Never expanded here: a
 *  label is an identifier, and the caller decides whether it is also prose.
 *
 *  `sport` is part of the one signature every reader shares, and this reader
 *  does not need it: the shape that names a pick is the shape, whatever the
 *  sport. Renamed with the underscore so the unused parameter is honest rather
 *  than deleted, since the callers and the tests both pass it. */
export function pickLabel(bundle: any, _sport: string): string | null {
  const pick = bundle?.pick;
  if (typeof pick === "string" && pick) return pick;
  if (pick && typeof pick === "object" && typeof pick.label === "string" && pick.label) {
    return pick.label;
  }
  return null;
}

/** The full name for a short label, when the bundle carries both.
 *
 *  Sites that abbreviate already carry the long name beside it
 *  (`home_team_full`), so this reads the bundle and never a table: a mapping
 *  invented here would be a roster this package does not own, and a stale one
 *  would put the wrong team's name on a verdict. No full name, no expansion —
 *  the label reads as the label. */
function fullTeamName(bundle: any, label: string): string {
  const home = bundle?.home_team ?? bundle?.team_home;
  const away = bundle?.away_team ?? bundle?.team_away;
  if (home === label) return bundle?.home_team_full ?? bundle?.team_home_full ?? home;
  if (away === label) return bundle?.away_team_full ?? bundle?.team_away_full ?? away;
  return label;
}

export function pickProbability(bundle: any): number | null {
  const p = bundle?.pick?.prob;
  return typeof p === "number" && Number.isFinite(p) && p >= 0 && p <= 1 ? p : null;
}

export function pickTiming(bundle: any, sport: string): Timing {
  const moment = MOMENT_OF[sport] ?? "kickoff";
  const label = pickLabel(bundle, sport);
  if (bundle?.pick_timing === "rebuilt") return { status: "rebuilt", moment };
  if (!label) return { status: "nopick", moment };
  if (bundle?.pick_timing === "unknown") return { status: "unverified", moment };
  return { status: "none", moment };
}

export function verdictSentence(bundle: any, sport: string): string | null {
  const label = pickLabel(bundle, sport);
  return label ? `${fullTeamName(bundle, label)} is the pick.` : null;
}