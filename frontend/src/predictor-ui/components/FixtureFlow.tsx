// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** FixtureFlow — instant, local, state-appropriate description.
 *
 *  Pure function of its props.  No useEffect, no fetch, no timer.
 *  Renders the game flow for the given sport and state.
 *
 *  **The pre-game prose is gone, and the instant block is why.** The block
 *  (`InstantBlock`) draws the timing, the verdict, the tiles, the bar and the
 *  record from the very same bundle; saying them here as sentences is what put
 *  one figure on a page in two places with two values. What is left here is the
 *  fixture's own name — a name, not a claim — and then only the sentences that
 *  carry news: the score and how it stands while the fixture is live, and the
 *  result and the pick's rightness once it is finished. A state with nothing to
 *  say renders an empty flow, which is the honest reading: the facts are above.
 *
 *  Honour assertions (must bite):
 *  - No line named that is not there (test on rendered strings, not keys).
 *  - No direction marker, no "the market disagrees", no derived percentage.
 *  - F1 draws no market-line tile and no split-bar market row.
 *  - Every sentence is true for every bundle that reaches it.
 */

import { pickLabel, pickProbability } from "../lib/bundleFacts";

export type FlowState = "pre-game" | "in-play" | "finished";

export interface FixtureFlowProps {
  sport: string;
  state: FlowState;
  bundle: any;
  /** The panel's request function for the AI summary.  The flow is a pure
   *  function of the facts and never calls it — it is accepted so the parent
   *  can pass one prop to both children. */
  request: () => Promise<any>;
}

/** Whole percent, matching the panel's own formatting. */
function pct(p: number): string {
  if (p <= 0.005) return "<1%";
  if (p >= 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}

/** Whether the bundle carries a market line to stand against. */
function hasLine(bundle: any): boolean {
  const line = bundle?.market_line;
  return line !== null && line !== undefined && line !== "";
}

/** The score, when the facts carry one. */
function scoreSentence(bundle: any): string | null {
  const score = bundle?.score;
  if (!score || typeof score.home !== "number" || typeof score.away !== "number") return null;
  const home = bundle?.home_team ?? "Home";
  const away = bundle?.away_team ?? "Away";
  return `The score is ${home} ${score.home}, ${away} ${score.away}.`;
}

/** The state of the game, for a sport with no score to show (F1). */
function stateSentence(sport: string, bundle: any): string | null {
  if (sport !== "f1") return null;
  const label = pickLabel(bundle, sport) ?? bundle?.driver;
  if (!label) return null;
  const prob = pickProbability(bundle);
  return prob !== null
    ? `The state of the race: the model's pick is ${label}, at ${pct(prob)} win probability.`
    : `The state of the race: the model's pick is ${label}.`;
}

/** How the score stands against the line, when there is both. */
function standingSentence(bundle: any): string | null {
  const score = bundle?.score;
  if (!score || typeof score.home !== "number" || typeof score.away !== "number") return null;
  if (!hasLine(bundle)) return null;
  const home = bundle?.home_team ?? "Home";
  const away = bundle?.away_team ?? "Away";
  const diff = score.home - score.away;
  const lead =
    diff > 0 ? `${home} lead by ${diff}` : diff < 0 ? `${away} lead by ${-diff}` : "The scores are level";
  return `${lead}; the market's line is ${bundle.market_line}.`;
}

/** The result, in the reader's own words. */
function resultSentence(sport: string, bundle: any): string | null {
  const result = bundle?.result;
  if (result === "draw") return "The result is a draw.";
  if (result === "home_win") return `The result is a win for ${bundle?.home_team ?? "the home side"}.`;
  if (result === "away_win") return `The result is a win for ${bundle?.away_team ?? "the away side"}.`;
  if (result === "home_cover") return `The result is a cover for ${bundle?.home_team ?? "the home side"}.`;
  if (result === "away_cover") return `The result is a cover for ${bundle?.away_team ?? "the away side"}.`;
  if (result) return `The result is ${result}.`;
  // F1 carries no result field — the pick's rightness is the result.
  if (sport === "f1") {
    const wasRight = bundle?.pick?.was_right;
    const label = pickLabel(bundle, sport) ?? bundle?.driver;
    if (typeof wasRight === "boolean" && label) {
      return wasRight
        ? `The result is a win for ${label}.`
        : `The result is not a win for ${label}.`;
    }
  }
  return null;
}

/** Whether the model's pick was right. */
function rightnessSentence(bundle: any): string | null {
  const wasRight = bundle?.pick?.was_right;
  if (typeof wasRight !== "boolean") return null;
  return wasRight
    ? "The pick rightness: the model's pick was right."
    : "The pick rightness: the model's pick was wrong.";
}

/** One row of the flow. `heading` rows are the fixture's own name. */
type Row = { text: string; heading?: boolean };

/** Build the flow's rows for the given sport and state. */
function buildRows(sport: string, state: FlowState, bundle: any): Row[] {
  const rows: Row[] = [];

  if (state === "pre-game") {
    // The fixture's name and nothing else. The pick, the probabilities, the
    // timing and the line are the block's, as figures, one of them.
    //
    // F1 names a driver rather than a home and an away side, so it has no name
    // row here and the pre-game flow is empty for that sport — deliberately. The
    // block states the pick; the flow adds nothing to it.
    const home = bundle?.home_team;
    const away = bundle?.away_team;
    if (home && away) rows.push({ heading: true, text: `${home} vs ${away}` });
  } else if (state === "in-play") {
    const score = scoreSentence(bundle);
    if (score) rows.push({ text: score });
    const st = stateSentence(sport, bundle);
    if (st) rows.push({ text: st });
    const standing = standingSentence(bundle);
    if (standing) rows.push({ text: standing });
  } else if (state === "finished") {
    const result = resultSentence(sport, bundle);
    if (result) rows.push({ text: result });
    const rightness = rightnessSentence(bundle);
    if (rightness) rows.push({ text: rightness });
  }

  return rows;
}

export function FixtureFlow({ sport, state, bundle }: FixtureFlowProps) {
  const rows = buildRows(sport, state, bundle);
  return (
    <div className="flex flex-col gap-2" data-testid="fixture-flow">
      {rows.map((row, i) =>
        row.heading ? (
          <h4 key={i} className="font-pr-display text-lg font-semibold text-pr-text">
            {row.text}
          </h4>
        ) : (
          <p key={i} className="text-sm leading-relaxed text-pr-text-dim">
            {row.text}
          </p>
        ),
      )}
    </div>
  );
}