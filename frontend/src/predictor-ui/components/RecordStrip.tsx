// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { record as fmtRecord } from "../fmt";

/**
 * The record strip: the two numbers the facts carry, and the proportion as
 * geometry.
 *
 * Spec §5a-bis, and this is the component that rule exists for. `record` is
 * `hits: 41, settled: 68`; the facts contain no `60.3`. So the text reads
 * `41/68` and the bar's width carries the proportion. Writing "60.3%" here would
 * be a third number this product invented by arithmetic on two real ones — the
 * same class of error as a model inventing a figure, and the reason the rule is
 * written down rather than left to whoever renders it.
 *
 * The accessible name says "41 of 68" for the same reason. A screen reader
 * cannot see a bar's width, so the numbers have to be in words; those words are
 * the facts' own, not a percentage derived from them.
 */
export function RecordStrip({
  label,
  hits,
  settled,
}: {
  label: string;
  hits: number | null;
  settled: number;
}) {
  const counts = fmtRecord(hits ?? Number.NaN, settled);
  const has = counts !== "—" && settled > 0;
  const share = has ? Math.max(0, Math.min(1, (hits ?? 0) / settled)) : 0;

  /* One reading of `record`, including the partial one, and the reason it is a
   * decision rather than a fall-out.
   *
   * `hits` and `settled` are two fields of ONE `record` object in the facts
   * bundle (`facts.py:29`; every fixture in the service's own tests carries both,
   * and `template.py:159` refuses to write a record row at all unless both are
   * there). So `hits: null, settled: 68` is not a record missing its numerator —
   * it is a record object this component cannot read, and the honest rendering
   * of "we cannot read this" is the dash, the same token `pct()` and
   * `fmt.record()` already use.
   *
   * **The trade-off, stated because it is real: this throws away a 68 the caller
   * did hand over, and `—/68` would show it.** Rejected, and the reason is the
   * same one that stops the bar being drawn 0% wide: `settled` comes out of the
   * very object whose `hits` we have just declined to trust, so quoting the
   * denominator while withholding the numerator asserts that one of the two is
   * sound when the only thing we know is that the object is not. There is also an
   * accessibility cost — "—/68" announces as "em dash slash sixty-eight", which
   * is exactly the run of glyphs the bar's own "41 of 68" accessible name exists
   * to avoid.
   *
   * What would change the ruling: the service emitting a partial record
   * deliberately, or the facts guaranteeing `settled` independently of `hits`.
   * Neither is true, so there is no partial here to be honest *about*, and the
   * dash is the whole of the truth. Pinned by a test rather than left to the
   * next reader of this file.
   */
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{label}</span>
      {has ? (
        <>
          <span
            role="img"
            aria-label={`${hits} of ${settled}`}
            className="flex h-1.5 w-full overflow-hidden rounded-pr bg-pr-panel-2"
          >
            <span data-testid="record-fill" className="h-full bg-pr-accent" style={{ width: `${share * 100}%` }} />
          </span>
          <span className="font-pr-display text-sm font-semibold text-pr-text">{counts}</span>
        </>
      ) : (
        // Nothing settled reads as a dash, never 0/0 — "0/0" is a claim about a
        // record that does not exist yet.
        <span className="font-pr-display text-sm font-semibold text-pr-text-faint">—</span>
      )}
    </div>
  );
}
