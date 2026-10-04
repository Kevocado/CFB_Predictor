// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/**
 * A presentational box score. It renders a structure the caller has already
 * built; it never fetches, never reorders, and does not know what a touchdown
 * is. If it decided ordering, the caller could not put a team's players in
 * depth-chart order, which is the one ordering that matters.
 */

export type BoxScoreRow = {
  key: string; // stable React key
  name: string;
  position: string; // display group label
  team: string; // abbreviation
  order: number; // caller-assigned; lower sorts first
  isStarter: boolean | null; // null = this sport has no real starter data
  values: (number | null)[]; // aligned to `columns`; null renders an em-dash
  emphasisIndex?: number; // which of `values` is the one number that shouts
};

export type BoxScoreColumn = {
  key: string;
  label: string;
  align?: "left" | "right";
  /** Screen-reader text for a row's cell, since the grid itself is terse. */
  describe?: (value: number | null, row: BoxScoreRow) => string;
};

export type BoxScoreGroup = {
  position: string;
  /** Already grouped, already ordered starters-then-bench. Rendered as given. */
  rows: BoxScoreRow[];
  /** Per-team subtotal for this position, or null when not wanted. */
  subtotals?: BoxScoreTotal[] | null;
};

export type BoxScoreTotal = {
  label: string; // e.g. "MIA total"
  values: (number | null)[]; // aligned to the same `columns`
  strong?: boolean; // the projected score: renders at display scale
};

export interface BoxScoreProps {
  columns: BoxScoreColumn[];
  groups: BoxScoreGroup[];
  /** Heading for the block, e.g. "Predicted box score". */
  title?: string;
  /** Which states exist across the rows, which decides the note. */
  starterLabel?: string;
  benchLabel?: string;
}

const EM_DASH = "—";

function cellText(value: number | null | undefined): string {
  // The finite check is not defensive padding. A consumer that computes a
  // value from missing data produces NaN, and NaN here would render as the
  // literal text "NaN" in a public table. Anything that is not a real number is
  // "no value", so it gets the em-dash like any other absence.
  if (value === null || value === undefined || !Number.isFinite(value)) return EM_DASH;
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/**
 * The three starter states, decided across every row rather than per row.
 *
 * A single null row in an otherwise-known game must not flip the whole table
 * to "projected", or the header would lie about the eleven rows that do have a
 * real starter. The header reports what the DATA knows, and a per-row tag
 * reports the row that does not.
 */
function starterState(groups: BoxScoreGroup[]): "known" | "projected" | "none" {
  let seenNull = false;
  let seenStarter = false;
  for (const group of groups) {
    for (const row of group.rows) {
      if (row.isStarter === null) seenNull = true;
      else if (row.isStarter) seenStarter = true;
    }
  }
  if (seenStarter) return "known";
  if (seenNull) return "projected";
  return "none";
}

function RowCells({
  row,
  columns,
}: {
  row: BoxScoreRow;
  columns: BoxScoreColumn[];
}) {
  return (
    <>
      {columns.map((column, i) => {
        const value = row.values[i] ?? null;
        // The describe hook exists because the grid is terse: "17.9" alone is
        // not a sentence. Without it a screen reader hears a bare number.
        const description = column.describe?.(value, row);
        return (
          <td
            key={column.key}
            className={
              (column.align === "left" ? "text-left " : "text-right ") +
              (row.emphasisIndex === i ? "font-semibold " : "") +
              "px-2 py-1.5 tabular-nums"
            }
            {...(description ? { "aria-label": description } : {})}
          >
            {cellText(value)}
          </td>
        );
      })}
    </>
  );
}

export function BoxScore({ columns, groups, title, starterLabel = "Starters", benchLabel = "Bench" }: BoxScoreProps) {
  const state = starterState(groups);
  const rowTotal = groups.reduce((n, g) => n + g.rows.length, 0);

  if (rowTotal === 0) return null;

  const nameWidth = 30;

  return (
    <section className="mt-5" data-testid="box-score">
      {title && (
        <h3 className="mb-1 text-sm text-pr-text-dim">
          {title}
          {state === "projected" && (
            <span className="ml-2 text-xs" data-testid="box-score-projected-note">
              Projected order — no depth-chart feed for this sport
            </span>
          )}
        </h3>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm" data-testid="box-score-table">
          <caption className="sr-only">
            {title ?? "Predicted box score"}
            {state === "projected" ? ", in projected order: this sport has no depth-chart feed" : ""}
          </caption>
          <thead>
            <tr className="border-b border-pr-rule text-[0.6875rem] uppercase tracking-wide text-pr-text-dim">
              <th scope="col" className={`w-[${nameWidth}%] px-2 py-1.5 text-left font-semibold`}>
                Player
              </th>
              {columns.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  className={
                    (column.align === "left" ? "text-left " : "text-right ") +
                    "px-2 py-1.5 font-semibold"
                  }
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>

          {groups.map((group) => (
            <tbody key={group.position}>
              <tr>
                <th
                  scope="colgroup"
                  colSpan={columns.length + 1}
                  className="bg-pr-panel px-2 py-1 text-left font-display text-xs font-bold uppercase tracking-wider text-pr-text"
                >
                  {group.position}
                </th>
              </tr>
              {group.rows.map((row) => (
                <tr
                  key={row.key}
                  data-testid="box-score-row"
                  data-starter={row.isStarter === null ? "projected" : row.isStarter ? "starter" : "bench"}
                  className="border-b border-pr-rule/60"
                >
                  <th scope="row" className="px-2 py-1.5 text-left font-normal">
                    <span>{row.name}</span>
                    {row.team && <span className="ml-1.5 text-xs text-pr-text-dim">{row.team}</span>}
                    {row.isStarter === null && (
                      <span className="sr-only"> — projected order, no depth-chart data</span>
                    )}
                  </th>
                  <RowCells row={row} columns={columns} />
                </tr>
              ))}
              {group.subtotals?.map((total) => (
                <tr key={total.label} data-testid="box-score-subtotal" className="border-t border-pr-rule">
                  <th scope="row" className="px-2 py-1.5 text-left font-semibold">
                    {total.label}
                  </th>
                  {total.values.map((value, i) => (
                    <td
                      key={columns[i]?.key ?? i}
                      className={
                        (columns[i]?.align === "left" ? "text-left " : "text-right ") +
                        "px-2 py-1.5 font-semibold tabular-nums " +
                        (total.strong ? "font-display text-base" : "")
                      }
                    >
                      {cellText(value ?? null)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>

      {state === "known" && (
        <p className="mt-1 text-xs text-pr-text-dim">
          {starterLabel} listed before {benchLabel.toLowerCase()}.
        </p>
      )}
    </section>
  );
}
