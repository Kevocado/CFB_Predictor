// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** FixtureExplainer — the panel, three states.
 *
 *  The default state is the facts plus the flow: the instant block above (timing,
 *  verdict, tiles, bar, record) and, under it, an instant local description built
 *  from what the site already has, with no request made. The AI summary is behind
 *  a button, so nothing is spent until a reader asks. When the button's request
 *  resolves, the summary swaps in over a block and a flow that never unmount —
 *  there is no frame in which the panel is empty. When the request fails, the
 *  facts and the flow are still on screen and the button offers a retry.
 *
 *  The block and the flow are always mounted. They are the states that are always
 *  true, and keeping them mounted is what "the flow does not flicker away" means:
 *  the summary appears above them, not instead of them after a gap.
 */
import { useState } from "react";
import { FixtureFlow, type FlowState } from "./FixtureFlow";
import { SummaryButton } from "./SummaryButton";
import { ExplainerPanel, type Explanation } from "./ExplainerPanel";
import { AI_PROMISE, InstantBlock } from "./InstantBlock";
import type { Moment } from "./StatusBadge";
import type { MarketTile } from "./KeyNumberTile";
import type { Segment } from "./ProbabilityBar";
import { barPick } from "../lib/panelFacts";

/** A summary as the service sends it: a v2 answer, plus the optional news date.
 *  The news date arrives with the summary body; when it is absent the element is
 *  omitted entirely, so the panel ships before the service sends the field. */
export type Summary = Explanation & { news_date?: string };

/** The site's own figures, drawn under the AI prose in the summary state.
 *
 *  The panel renders no figure of its own: the summary carries the words and
 *  the site carries the numbers. A site that already draws tiles, a bar, a
 *  record strip or a player list beside its panel passes them here and the
 *  summary state keeps them; a site with nothing to draw passes nothing and
 *  the summary renders bare. Every field is optional for exactly that reason.
 *
 *  `segments` does double duty: it draws the bar, and it is what the answer's
 *  pick is joined against. The service words a pick in its own vocabulary
 *  ("Ravens win", the shape PL ships) while a site labels its segments in
 *  theirs ("Ravens"), so the pick is translated through `barPick` before it
 *  reaches the panel — exactly label matches pass through, a trailing " win"
 *  is dropped onto a matching segment, and anything unplaceable is returned
 *  unchanged so the bar fails closed (nothing accented) rather than accenting
 *  the nearest segment to a claim nobody made.
 */
export interface FixtureExtras {
  tiles?: MarketTile[];
  segments?: Segment[];
  legend?: Segment[];
  record?: { label: string; hits: number | null; settled: number };
  players?: { name: string; projection: string }[];
  moment?: Moment;
}

/** The panel's three states. `unavailable` differs from `flow` only in what the
 *  button offers: a retry, rather than a first ask. */
type PanelState = "flow" | "summary" | "unavailable";

export interface FixtureExplainerProps {
  sport: string;
  state: FlowState;
  bundle: any;
  /** The panel's request function for the AI summary. Passed to both children;
   *  the flow never calls it. */
  request: () => Promise<unknown>;
  /** The site's own figures for the summary state. Absent by default. */
  extras?: FixtureExtras;
  /** What the button adds, said before it is pressed. Defaults to the package's
   *  own words (`AI_PROMISE`); a site may name its own surface instead. */
  promise?: string;
}

/** The AI prose, rendered from the summary the button fetched. Reuses the
 *  existing panel so the verdict, band, factors and footer are the same ones
 *  every site already shows, with the site's own figures underneath. */
function SummaryView({ summary, extras }: { summary: Summary; extras?: FixtureExtras }) {
  const segments = extras?.segments ?? [];
  const pick = summary.pick ? barPick(summary.pick, segments) : undefined;
  const data = pick ? { ...summary, pick } : summary;
  return (
    <div className="flex flex-col gap-3" data-testid="fixture-summary">
      <ExplainerPanel
        data={data}
        loading={false}
        error={false}
        onRetry={() => {}}
        // No tiles, bar, legend or record: the instant block above already renders
        // each of them, and a second copy is the overlap this phase removes.
        players={extras?.players}
        moment={extras?.moment}
      />
      {summary.news_date && (
        <p className="text-xs text-pr-text-faint" data-testid="news-date">
          Team news from {summary.news_date}
        </p>
      )}
    </div>
  );
}

export function FixtureExplainer({ sport, state, bundle, request, extras, promise = AI_PROMISE }: FixtureExplainerProps) {
  const [panelState, setPanelState] = useState<PanelState>("flow");
  const [summary, setSummary] = useState<Summary | null>(null);

  const handleSummary = (s: Explanation) => {
    // The button has already validated the shape; the news date rides along.
    setSummary(s as Summary);
    setPanelState("summary");
  };

  const handleUnavailable = () => {
    setPanelState("unavailable");
  };

  return (
    <div className="flex flex-col gap-3" data-testid="fixture-explainer">
      {/* Facts first, interpretation after: the block is the finished "what", so the
          AI summary reads as what it adds and never as a second copy of the page. */}
      <InstantBlock sport={sport} bundle={bundle} extras={extras} />
      {panelState === "summary" && summary && <SummaryView summary={summary} extras={extras} />}
      <FixtureFlow sport={sport} state={state} bundle={bundle} request={request} />
      {panelState !== "summary" && (
        <div className="flex flex-col gap-1">
          <SummaryButton
            request={request}
            onSummary={handleSummary}
            onUnavailable={handleUnavailable}
            label={panelState === "unavailable" ? "Try again" : "Get the AI summary"}
          />
          <p className="max-w-[70ch] text-xs text-pr-text-faint" data-testid="ai-promise">{promise}</p>
        </div>
      )}
    </div>
  );
}
