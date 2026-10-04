// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** Shared v2 panelFacts adapter.
 *
 *  One function, two call sites.  Each call site passes an input tagged with
 *  `kind: "PL"` or `kind: "SP"`.  The function dispatches on that tag and
 *  returns a union output type.
 *
 *  PL input: `{ kind: "PL"; fixture: PLFixture }`
 *  SP input: `{ kind: "SP"; game: SPGame; prediction: SPPrediction | null }`
 *
 *  PL output: `{ tiles: MarketTile[]; segments: Segment[]; legend: Segment[] | undefined }`
 *  SP output: `{ tiles: MarketTile[]; segments: Segment[]; legend: undefined }`
 *
 *  The input shapes are minimal structural interfaces declared here, not
 *  imports: the two sites' `GameSummary` types differ in fields this adapter
 *  never reads, and a shared library must not depend on either site's source.
 *  A caller's richer object assigns structurally.
 *
 *  `news_date` is an optional field on the output, read from the service; when
 *  absent the footer element is omitted.  This task does not depend on Plan 1.
 *
 *  The SP branch is a faithful port of Sports' `src/lib/panelFacts.ts` (the
 *  spec's rollout rule: every assertion in the site's tests must hold here).
 *  In particular the spread tile keeps the home-frame convention the field
 *  arrives in — `spread_line` is the home team's expected margin and
 *  `predicted_margin` is home minus away, so both numbers take one sign flip
 *  and their disagreement survives in every sign combination — and the moneyline
 *  tile carries the leading side formatted by the family's `pct`, never a raw
 *  decimal. An earlier draft of this branch showed the unsigned line beside an
 *  absolute discrepancy instead; the site's sign-combination matrix pins the
 *  double-flip form, so the draft's form is the one that changed.
 */
import { pct, signed, spread, stat } from "../fmt";
import type { MarketTile, PickRef, Segment } from "../index";

/** One side of PL's three-way market: the model's probability and the market's
 *  implied figure side by side. Either may be absent; the adapter carries each
 *  as far as the facts carry it and derives nothing. */
export interface PLEdge {
  prob: number | null;
  implied?: number | null;
}

/** The fields of a PL fixture this adapter reads. */
export interface PLFixture {
  team_home: string;
  team_away: string;
  home_win: PLEdge | null;
  draw: PLEdge | null;
  away_win: PLEdge | null;
  predicted_total_goals: number | null;
  btts_yes_prob: number | null;
}

/** The fields of a Sports game this adapter reads. All lines optional: callers
 *  pass their own `GameSummary`, whose lines are optional (`spread_line?`),
 *  and an absent line must read as absent, not as zero. */
export interface SPGame {
  home_team: string;
  away_team: string;
  spread_line?: number | null;
  total_line?: number | null;
}

/** The fields of a Sports prediction this adapter reads. */
export interface SPPrediction {
  home_win_prob?: number | null;
  away_win_prob?: number | null;
  predicted_margin?: number | null;
  predicted_total?: number | null;
}

/** PL three-way result shape — the adapter keeps `legend` when `implied` is present. */
interface PLInput {
  kind: "PL";
  fixture: PLFixture;
}
interface PLOutput {
  tiles: MarketTile[];
  segments: Segment[];
  legend: Segment[] | undefined;
}

/** Sports two-way moneyline + spread + total shape. */
interface SPInput {
  kind: "SP";
  game: SPGame;
  prediction: SPPrediction | null;
}
interface SPOutput {
  tiles: MarketTile[];
  segments: Segment[];
  /** Sports never carries `implied`, so legend is always undefined here. */
  legend: undefined;
}

/** Union output — the callee selects the branch that matches its input. */
export type PanelFactsOutput = PLOutput | SPOutput;

/** A probability is only a probability if it is one. A missing field is not 0. */
function prob(x: number | null | undefined): number | null {
  return typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1 ? x : null;
}

function num(x: number | null | undefined): number | null {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

/** Shared adapter — dispatch on the `kind` tag. */
export function panelFacts(
  input: PLInput
): PLOutput;
export function panelFacts(
  input: SPInput
): SPOutput;
export function panelFacts(
  input: PLInput | SPInput
): PanelFactsOutput {
  if (input.kind === "PL") {
    const { fixture } = input;
    const tiles: MarketTile[] = [];
    const segments: Segment[] = [];
    const legend: Segment[] = [];
    if (!fixture) return { tiles, segments, legend };

    // The three outcomes, in the order a fixture is read. The draw is its own
    // segment: a three-way market rendered as two ways is a lie by layout.
    // Each side is an edge object carrying the model's probability and the
    // market's implied figure side by side; either may be absent.
    const sides = [
      { key: "home_win", edge: fixture.home_win },
      { key: "draw", edge: fixture.draw },
      { key: "away_win", edge: fixture.away_win },
    ] as const;
    const known = sides
      .map(({ key, edge }) => ({ key, prob: prob(edge?.prob), implied: num(edge?.implied) }))
      .filter(
        (e): e is { key: "home_win" | "draw" | "away_win"; prob: number; implied: number | null } =>
          e.prob !== null,
      );

    if (known.length) {
      // Labels are the teams where we have them, and the bare side otherwise.
      for (const e of known) {
        const label =
          e.key === "home_win"
            ? fixture.team_home
            : e.key === "away_win"
              ? fixture.team_away
              : "Draw";
        segments.push({ label, prob: e.prob, market: "result" });
        // The legend is carried as far as the market's `implied` carries it
        // and no further: per outcome, including the draw. Nothing is derived
        // to complete the set, and whether the row draws is the bar's decision
        // (`covers()`), not this mapping's.
        if (e.implied !== null) {
          legend.push({ label, prob: e.implied, market: "result" });
        }
      }

      // The tile carries the LEADING outcome, because a tile showing the
      // draw's 26% answers a question nobody asked; the bar beneath carries
      // all three.
      const lead = [...known].sort((a, b) => b.prob - a.prob)[0];
      const leadLabel =
        lead.key === "home_win"
          ? fixture.team_home
          : lead.key === "away_win"
            ? fixture.team_away
            : "Draw";
      tiles.push({
        market: "result",
        label: "result",
        value: pct(lead.prob),
        sub: `win · ${leadLabel}`,
      });
    }

    const total = num(fixture.predicted_total_goals);
    if (total !== null) {
      tiles.push({
        market: "total_goals",
        label: "total goals",
        value: stat(total),
      });
    }

    const btts = prob(fixture.btts_yes_prob);
    if (btts !== null) {
      tiles.push({
        market: "btts",
        label: "both score",
        value: pct(btts),
      });
    }

    return { tiles, segments, legend };
  }

  // --- Sports two-way moneyline + spread + total -------------------------
  // Ported from Sports' `src/lib/panelFacts.ts`: same inputs, same outputs.
  // The four rules that port with it:
  //
  //  - **The spread tile names the home team, and puts BOTH of its numbers in
  //    the home team's own convention.** `spread_line` arrives in the home
  //    frame and `predicted_margin` is home minus away, so the disagreement is
  //    a subtraction in that one frame and nothing else.
  //  - **A market the data does not carry is not passed at all.** No
  //    placeholder, no dash, no zero.
  //  - **The segments carry the `market` key**, so a factor that references
  //    "moneyline" highlights the bar rather than being a label nobody can
  //    connect.
  //  - **`fmt` owns every number.** `pct`, `stat`, `signed` and `spread` are
  //    the family's one place for rounding, signs and team names.
  if (input.kind === "SP") {
    const { game, prediction } = input;
    const tiles: MarketTile[] = [];
    const segments: Segment[] = [];
    if (!prediction) return { tiles, segments, legend: undefined };

    const home = prob(prediction.home_win_prob);
    const away = prob(prediction.away_win_prob);

    if (home !== null && away !== null) {
      // Home first, because that is the order a fixture is written in and the
      // order the family has always used for a two-way bar.
      segments.push(
        { label: game.home_team, prob: home, market: "moneyline" },
        { label: game.away_team, prob: away, market: "moneyline" },
      );
      // The tile carries the LEADING side, because a tile whose value is the
      // underdog's number answers a question nobody asked. The bar below
      // already carries both.
      const lead = home >= away
        ? { team: game.home_team, p: home }
        : { team: game.away_team, p: away };
      tiles.push({
        market: "moneyline",
        label: "moneyline",
        value: pct(lead.p),
        sub: `win · ${lead.team}`,
      });
    }

    // The spread tile needs BOTH the market's line and the model's own margin:
    // a tile showing only the line would be the market's number, and a tile
    // showing only the margin would be a figure with nothing to disagree with.
    // The disagreement is the point.
    //
    // **THE FRAME: the home team's, because that is the field arrives in.**
    // `spread_line` is ALREADY the home team's own line and `predicted_margin`
    // is home minus away, so the tile names the HOME team, always, and both
    // numbers take one sign flip (`spread(team, -line)`, `signed(-margin)`):
    // flipping one without the other leaves the tile doing arithmetic the
    // reader can see and get a different answer from. The double flip keeps
    // the rendered gap equal to `margin - line` in every sign combination,
    // which the sign matrix below pins.
    const margin = num(prediction.predicted_margin);
    const line = num(game.spread_line);
    if (margin !== null && line !== null) {
      tiles.push({
        market: "spread",
        label: "spread",
        value: spread(game.home_team, -line),
        sub: `model ${signed(-margin)}`,
      });
    }

    const total = num(prediction.predicted_total);
    const totalLine = num(game.total_line);
    if (total !== null && totalLine !== null) {
      tiles.push({
        market: "total",
        label: "total",
        value: stat(total),
        sub: `total pts · line ${stat(totalLine)}`,
      });
    }

    return { tiles, segments, legend: undefined };
  }

  // Fallback: return empty output
  return { tiles: [], segments: [], legend: undefined };
}

/** What PL's `/facts` calls the two sides. `match_pick` in the PL API's facts
 *  module words the pick `"<team> win"`; this is the only suffix the family's
 *  explainers are known to add to a team name, and it is the whole reason the
 *  bar needs translating at all. */
const WIN_SUFFIX = " win";

/** The answer's pick, in the vocabulary the calling site's bar is drawn in.
 *
 *  Ported from Sports' `src/lib/panelFacts.ts` with the site-specific comment
 *  kept where it carries the warning: the bar joins the pick to a segment **by
 *  label**, and an exact label passes through, a trailing " win" is dropped
 *  and the bare name re-joined to a segment, and anything unplaceable is
 *  returned **unchanged** so the bar fails closed — nothing accented — rather
 *  than accenting the nearest segment to a claim nobody made.
 *
 *  Takes a pick rather than a pick-or-nothing on purpose: whether there is a
 *  pick is the caller's question to ask before it gets here.
 */
export function barPick(pick: PickRef, segments: Segment[]): PickRef {
  if (segments.some((s) => s.label === pick.label)) return pick;
  if (!pick.label.endsWith(WIN_SUFFIX)) return pick;
  const bare = pick.label.slice(0, -WIN_SUFFIX.length);
  const match = segments.find((s) => s.label === bare);
  return match ? { ...pick, label: match.label } : pick;
}
