// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";

/**
 * One market's answer, figure-led: the model's number large, and what the market
 * itself says directly beneath it.
 *
 * Deliberately not `StatTile`, which leads with the label. The label here IS the
 * answer, so it has to be the largest thing on the surface — and the tile's job
 * is the *comparison* between the model and the market, which is why `sub` sits
 * under the figure rather than above it. A tile that showed only the model's
 * number would be a vanity metric.
 *
 * A tile whose market the facts do not carry renders **nothing at all**: no
 * dashed box, no zero, no placeholder. Spec §6 says an absent market renders
 * nothing, and a dash would read as "we looked and there is nothing here", which
 * is a different claim from "there is no such market".
 */
export type MarketTile = {
  /** The facts' own `market` key. A factor's `key` refers to this, which is what
   *  makes the §13c highlight a lookup rather than a second naming scheme. It is
   *  an identifier, and it is used for the `data-testid` and the highlight only —
   *  never as something to read. */
  market: string;
  /** The words for this market, for a reader. Never a key: `btts` is what the
   *  facts call it, "Both teams score" is what a reader needs. */
  label: string;
  /** The figure. A string in every adapter that ships one today; typed wide
   *  because the PL branch still passes the raw decimal it has always passed
   *  and the renderer prints either. PL's own tests already expect formatted
   *  strings ("48%"), so the PL rollout narrows this back to `string` when it
   *  formats that branch — this widening is the transitional state, not the
   *  contract. */
  value: string | number;
  /** The market's own line beneath the figure, when the facts carry one. It
   *  stands in for `label` when present, so it has to read as a phrase too. */
  sub?: ReactNode;
  /** A colour for the market's own identity, where one exists. */
  tint?: string;
};

export function KeyNumberTile({ tile, highlighted }: { tile: MarketTile; highlighted?: boolean }) {
  // `== null`, not a falsiness test. `0` is a real figure: a 0-0 draw, a total of
  // zero, a probability the model has rounded to nothing. `!tile.value` hid all
  // three, and it did so silently — the tile simply was not on the page, which
  // reads as "no market here" rather than "the model said zero".
  //
  // `""` is still treated as absent, deliberately: the adapters produce a
  // formatted string for a figure they have, and an empty one means the figure
  // was missing upstream. Only the falsy NUMBER is a value.
  if (tile.value === null || tile.value === undefined || tile.value === "") return null;
  const { market, value, sub, tint } = tile;
  return (
    <dl
      data-testid={`tile-${market}`}
      data-highlighted={highlighted ? "true" : "false"}
      className={`flex min-w-0 flex-col gap-0.5 rounded-pr border px-3 py-2.5 transition-colors duration-150 ${
        highlighted ? "border-pr-accent bg-pr-panel-2" : "border-pr-rule bg-pr-panel"
      }`}
    >
      <dd
        className="font-pr-display text-2xl font-semibold leading-none text-pr-text"
        style={tint ? { color: tint } : undefined}
      >
        {value}
      </dd>
      {/* `sub` when the site gave one, otherwise the market's LABEL — never
       *  `market`. That fallback was the defect: `market` is the key the facts
       *  use to join on ("btts", "total_goals", "moneyline"), so a tile that
       *  carried a label and no sub line showed the reader a machine key in the
       *  one slot on the tile reserved for words — and it did it silently,
       *  because `label` was required, was accepted, and was never rendered
       *  anywhere. A key printed where a label belongs is the general shape of
       *  this bug; the words are the fix and they come from the caller's
       *  `label`, which is what that field is for. */}
      <dt className="text-xs leading-snug text-pr-text-dim">{sub ?? tile.label}</dt>
    </dl>
  );
}
