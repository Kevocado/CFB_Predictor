// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { useEffect, useId, useState } from "react";
import { ErrorState, Skeleton } from "./States";
import { StatusBadge, type Moment } from "./StatusBadge";
import { BandChip, PanelHeading, type Band } from "./ExplainerVerdict";
import { KeyNumberTile, type MarketTile } from "./KeyNumberTile";
import { FactorList, type Factor } from "./FactorList";
import { RecordStrip } from "./RecordStrip";
import { ProbabilityBar, type PickRef, type Segment } from "./ProbabilityBar";

/** What a v2 answer looks like: a verdict, a computed band, and factors that
 *  *reference* a market by key rather than carrying a figure.
 *
 *  `pick` is absent when there is no pick, which is the case the panel has to
 *  honour twice: no bar segment is accented, and nothing says "for the pick". */
export type Verdict = { verdict: string; band: Band; factors: Factor[]; pick?: PickRef };

/** Everything both shapes of answer carry. Exported so a caller that *builds* an
 *  answer can name the v2 arm (`Common & Verdict`) rather than the union: on
 *  the union, `.factors` does not exist, which is a type error at every call
 *  site that legitimately has a v2 answer in hand. */
export type Common = {
  /** "llm" means a model wrote these words; "template" means the site's own
   *  copy did, from the same numbers. The panel never blurs the two. */
  source: "llm" | "template";
  model: string;
  generated_at: string;
  /** The sport, so the panel words the rebuilt status for that moment. */
  sport: string;
  /** When the pick itself was made. "rebuilt" means the model was asked again
   *  after the event had begun, so the pick is shown but never counted.
   *  "unknown" means the schedule never supplied the start time, so the pick
   *  cannot be shown as made before the start. This
   *  travels with the answer rather than being read out of the prose: neither
   *  the model nor the template is a reliable source for a status label. */
  pick_timing: "pre_kickoff" | "rebuilt" | "none" | "unknown";
};

export type Explanation = Common & Verdict;

/** The moment an F1 pick has to beat is the session, not a kick-off. */
const MOMENT_OF: Record<string, Moment> = { f1: "the session", nba: "tip-off" };

/** What "after the moment" means in a sentence. StatusBadge owns the label;
 *  this owns why the pick is not being counted. */
const STARTED: Record<Moment, string> = {
  kickoff: "after the game started",
  "tip-off": "after the game started",
  "the session": "after the session started",
};

/** How long ago a summary was written, in the reader's own units. A summary
 *  from 20 seconds ago is "just now", not "0 min ago" — nobody writes that. */
function ago(iso: string, now: number): string {
  const then = Date.parse(iso);
  // Unparseable, or a clock skew so large the age would be negative. The
  // caller treats an empty age as "no age to quote" and falls back to the
  // template footer, rather than claiming a recency it cannot compute.
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((now - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} ${hours === 1 ? "hr" : "hrs"} ago`;
  const days = Math.round(hours / 24);
  return `${days} ${days === 1 ? "day" : "days"} ago`;
}

/**
 * The panel's footer is the honesty contract, and it is the one claim in the
 * product that has no fact behind it: it asserts who wrote these words. So it
 * fails CLOSED. The model-written footer is rendered only when the response
 * actually carries the two things that footer asserts — the model and a
 * timestamp we can read — and anything else (a templated summary, a missing
 * field, a renamed value on the wire, a clock we cannot parse) falls back to the
 * sentence that claims nothing. An "AI" label with a blank model beside
 * it, or with no age, is worse than no label: it looks verified and is not.
 */
function footer(data: Explanation, now: number): string {
  const TEMPLATE = "Summary written from the model's numbers";
  if (data.source !== "llm" || !data.model) return TEMPLATE;
  const age = ago(data.generated_at, now);
  if (!age) return TEMPLATE;
  return `AI summary of the model's numbers · ${data.model} · ${age}`;
}

const toggleClass =
  "font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-accent underline-offset-4 transition-colors hover:text-pr-text focus-visible:text-pr-text";

export function ExplainerPanel({
  data,
  loading,
  error,
  onRetry,
  collapsed = false,
  moment,
  tiles = [],
  segments,
  legend,
  expandable = false,
  record,
  players,
}: {
  data: Explanation | null;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
  collapsed?: boolean;
  /** Overrides the moment the sport implies. Sites pass their own so the
   *  status reads in the words that site already uses everywhere else. */
  moment?: Moment;
  /** One per market the facts carry. An absent market is simply not passed, and
   *  `KeyNumberTile` renders nothing for an empty one — never a dash. */
  tiles?: MarketTile[];
  segments?: Segment[];
  /** The market's own split, rendered only when it covers every outcome
   *  `segments` has (§13b). */
  legend?: Segment[];
  /** For a surface that can be too narrow for a row of figures — the market
   *  row's labels collapse behind a real `aria-expanded` button, the same shape
   *  `FactorList` uses for a clamped sentence. Off by default, so a wide panel
   *  never hides text behind a control that adds nothing.
   *
   *  Opt-in, deliberately, and `ProbabilityBar`'s prop comment carries the ruling
   *  and the three CSS answers that were measured against §6a and §13c and
   *  rejected. The short version: a CSS-only fallback either breaks the 12px floor
   *  or withholds the figures, and the failure mode of a site that forgets this is
   *  a collision at 260px, which a screenshot catches. */
  expandable?: boolean;
  record?: { label: string; hits: number | null; settled: number };
  /** NFL's top player projections (§6). A list the facts already carry, so it
   *  costs the panel nothing to show. */
  players?: { name: string; projection: string }[];
}) {
  const [opened, setOpened] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  // useId, not a literal: two panels on one page (a fixture and a track-record
  // row, say) would otherwise share an id, and aria-labelledby would point at
  // the wrong one.
  const headingId = useId();
  const showBody = !collapsed || opened;
  const v2 = !!data && "factors" in data;

  // Escape clears a highlight, because a highlight that cannot be dismissed is
  // a stuck state and a keyboard reader has no other way out of it.
  useEffect(() => {
    if (!highlighted) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setHighlighted(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [highlighted]);

  if (loading) return <Skeleton label="Writing the summary…" />;
  if (error) {
    // The panel cannot promise the numbers below are unaffected: the summary
    // may have failed because the same fetch that serves them did. Say what is
    // true — the summary is missing, here is another go, the rest is below.
    return (
      <ErrorState
        message="The summary didn't come through. Try again, or carry on with the numbers below."
        onRetry={onRetry}
      />
    );
  }
  if (!data) return null;

  const now = Date.now();
  const when = moment ?? MOMENT_OF[data.sport] ?? "kickoff";
  const rebuilt = data.pick_timing === "rebuilt";
  // An unverified pick shows no band either: every band word is a confidence
  // claim, and "leaning" on a pick the facts cannot place in time is still
  // one. Withheld like `rebuilt`, for the same reason — hiding it fails
  // closed — except nothing was rebuilt, so the badge says so instead.
  const unverified = data.pick_timing === "unknown";

  /* §13c's linkage is a LOOKUP, and this is the half of it that was missing: a
   * factor's `key` is only a reference to a figure if some figure carries it.
   *
   * The key used to be forwarded whatever it was, and `dim` then dimmed every
   * segment whose `market` was not the highlight — so a key that matched nothing
   * lit nothing and dimmed everything, which is the one outcome of a control that
   * is worse than either alternative. It is not a rare edge: `template.py` always
   * emits `record` (`:164`) and pads with `context` (`:178`/`:184`), and neither
   * is a market, so on a no-pick panel *every* row is unlinkable and every click
   * faded the whole panel with nothing lit. §2's rule is that a value on screen
   * comes from something real or is absent; the de-emphasis is a value, and it was
   * being derived from a reference that pointed at nothing.
   *
   * A key with no figure under it therefore clears the highlight instead of
   * setting one. That is not a row that does nothing: every row means "light the
   * figure this row is about, or clear the light if there is none to light", so
   * an unresolvable row turns a light off, which is a change the reader can see
   * and undo by pressing it again. The alternative — rendering such a row as
   * plain text, so it is not a control at all — was rejected: a reader cannot
   * tell which of their rows have figures, and the spec's §6 item 4 says a *why*
   * row is "still a row like any other: selectable, and expandable". */
  const linkable = (key: string) =>
    tiles.some((t) => t.market === key) || !!segments?.some((s) => s.market === key);
  const selectFactor = (key: string) =>
    setHighlighted((was) => (linkable(key) ? (was === key ? null : key) : null));

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-3">
      <PanelHeading id={headingId}>In plain English</PanelHeading>

      <div className="flex items-start justify-between gap-3">
        <p className="min-w-0 max-w-[70ch] text-lg font-medium leading-snug text-pr-text">
          {data.verdict}
        </p>
        {/* A rebuilt pick shows NO band (spec §13e), and the decision is this one
            line. The band is still computed by `band_for` and still in the
            response — withholding a value the panel cannot act on is a rendering
            decision, and keeping it computed keeps it available to whatever reads
            the row next. It is withheld rather than softened because the number it
            describes came from a model asked after the event began: it had seen
            the score, so the band measures confidence in a number produced with
            the answer already known, and the two lines below already tell the
            reader the pick was made then. "STRONG" beside that asks the reader to
            resolve a contradiction this panel created. Hiding it fails closed,
            which is the direction the footer already fails in.

            Note this survives the rule change and the band withholding does not:
            the track record counts the pick (spec
            2026-10-01-track-record-counts-every-pick), and a confidence word
            about a number produced with the answer already known is still a
            claim the panel cannot make honestly. The old *reason* for hiding it
            — that the reader was being told the pick was held out of the
            record — no longer holds, and the reason above is what replaced it.
            The withdrawn wording is quoted in `countedCopy.test.tsx`, which
            guards against it returning. */}
        {v2 && !rebuilt && !unverified && <BandChip band={data.band} />}
      </div>

      {rebuilt && (
        <p className="flex max-w-[70ch] flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="rebuilt" moment={when} />
          <span>
            This pick was made {STARTED[when]}. Counted in the track record like any other pick.
          </span>
        </p>
      )}

      {unverified && (
        <p className="flex max-w-[70ch] flex-wrap items-center gap-2 text-sm text-pr-text-dim">
          <StatusBadge status="unverified" moment={when} />
          <span>
            The schedule did not provide the start time, so this pick cannot be shown as made before the start.
          </span>
        </p>
      )}

      {collapsed && (
        <button
          type="button"
          onClick={() => setOpened((was) => !was)}
          aria-expanded={showBody}
          className={`mt-1 self-start underline ${toggleClass}`}
        >
          {showBody ? "Hide the race story" : "Read the race story"}
        </button>
      )}

      {showBody && v2 && (tiles.length > 0 || segments) && (
        <div className="flex flex-col gap-3">
          {tiles.length > 0 && (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {tiles.map((tile) => (
                <KeyNumberTile key={tile.market} tile={tile} highlighted={highlighted === tile.market} />
              ))}
            </div>
          )}
          {segments && segments.length > 0 && (
            <ProbabilityBar
              segments={segments}
              legend={legend}
              minSegmentPx={2}
              expandable={expandable}
              // The answer's pick, never worked out here. Absent, the bar accents
              // nothing, which is the honest rendering of a bar with no pick.
              pick={v2 ? data.pick : null}
              highlightKey={highlighted}
              // A segment's own `market`, so it resolves by construction — it is
              // this figure's key. The guard above exists for the other caller.
              onSegmentFocus={(_, market) => market && setHighlighted(market)}
            />
          )}
        </div>
      )}

      {showBody && v2 && (
        <div className="flex flex-col gap-2 border-t border-pr-rule pt-3">
          <h4 className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
            Why it matters
          </h4>
          <FactorList
            factors={data.factors}
            onSelect={selectFactor}
            // The row that asked for the light is the row that is pressed. Not
            // passed before, so pressing a *why* row changed the tiles and the bar
            // and left the row itself looking exactly as it had.
            highlighted={highlighted}
            expanded={expanded}
            onToggle={(i) => setExpanded((was) => (was === i ? null : i))}
          />
        </div>
      )}

      {/* The record strip and the player list are panel EXTRAS (§6), not
          factors, so they are not the v2 body's business. They render for any
          answer that carries them: every v2 panel once silently dropped its
          record, and no test asserted it, because the component test rendered
          RecordStrip directly and never through the panel. */}
      {showBody && (record || players) && (
        <div className="flex flex-wrap items-start gap-x-8 gap-y-3 border-t border-pr-rule pt-3">
          {players && players.length > 0 && (
            <div className="flex min-w-0 flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">Top projections</span>
              <ul className="flex flex-col gap-0.5">
                {players.slice(0, 3).map((pl) => (
                  <li key={pl.name} className="flex items-baseline justify-between gap-4 text-xs">
                    <span className="truncate text-pr-text">{pl.name}</span>
                    <span className="shrink-0 tabular-nums text-pr-text-dim">{pl.projection}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {record && <RecordStrip label={record.label} hits={record.hits} settled={record.settled} />}
        </div>
      )}

      <p className="mt-1 text-xs text-pr-text-faint">{footer(data, now)}</p>
    </section>
  );
}
