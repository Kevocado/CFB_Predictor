// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** The confidence word, as a word.
 *
 *  `band` arrives from the service already computed from `pick.prob` (spec
 *  §13a), and this component **renders what it is given and never recomputes
 *  one**. The thresholds live in one Python function; a second copy in
 *  TypeScript is a second thing to keep right, for no benefit, and the whole
 *  point of computing it server-side is that a model cannot call a 52% pick
 *  strong.
 *
 *  It is a chip rather than a coloured bar because a colour here would be a
 *  confidence nobody can check, and the words are what make it checkable. Uppercase
 *  and tracked out, matching the family’s status labels.
 */
export type Band = "leaning" | "moderate" | "strong";

const WORDS: Record<Band, string> = {
  leaning: "Leaning",
  moderate: "Moderate",
  strong: "Strong",
};

/**
 * The chip, or **nothing at all** when there is no band to put in it.
 *
 * **It fails closed, and it is the only thing in the panel that did not.** A body
 * that arrives without a `band` — or carrying a word this build has never heard of
 * — used to render `WORDS[band] ?? WORDS.moderate`, so it said **"Moderate"**: a
 * specific claim about how much to trust this pick, derived from nothing and
 * defaulted into existence. That is the whole thing this component exists to stop
 * happening, forty lines of contract away in `band_for`, which exists so a model
 * cannot call a 52% pick strong. Defaulting it here puts the same invention back
 * through the front door for any body the type system promised but the wire did
 * not.
 *
 * Every other uncertainty in this panel already fails closed, which is what makes
 * this one stand out rather than match the house style: `footer()` drops the "AI"
 * label on a blank model or an unparseable timestamp, `KeyNumberTile` returns
 * `null`, `RecordStrip` shows a dash, `pct()` returns "—". A band chip that
 * renders nothing leaves the verdict line exactly as it was, which is the honest
 * rendering of a confidence nobody supplied: the panel has a sentence and a
 * number, and no claim about how much to trust either.
 *
 * The prop is widened rather than kept as `Band` because the renderer is handed
 * whatever the service sent, and the type is a promise about the service rather
 * than a guarantee about the wire. `WORDS[band]` is `undefined` for a value
 * outside the union, so the guard covers the renamed value as well as the absent
 * one.
 */
export function BandChip({ band }: { band?: Band | null }) {
  const word = band == null ? undefined : WORDS[band];
  if (!word) return null;
  return (
    <span
      data-testid="band-chip"
      className="inline-flex shrink-0 items-center whitespace-nowrap rounded-pr border border-pr-rule bg-pr-panel px-2 py-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim"
    >
      {word}
    </span>
  );
}

/** The panel's own title. A real heading, so the panel is a landmark a screen
 *  reader can navigate to — not a small label sitting above a bigger one, which
 *  is the eyebrow the family does not use anywhere. */
export function PanelHeading({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h3 id={id} className="font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
      {children}
    </h3>
  );
}
