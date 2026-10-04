// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { Fragment, useState, type ReactNode } from "react";

export type Column<T> = {
  key: string;
  label: string;
  /** null = missing: renders "—" and sorts last in both directions. */
  value: (row: T) => number | string | null;
  /** Defaults to the value itself. */
  render?: (row: T) => ReactNode;
  numeric?: boolean;
  tooltip?: string;
  /** First-click direction. Numbers default to high-first; pass "asc" where lower is better. */
  firstDir?: "asc" | "desc";
};

type Sort = { key: string; dir: "asc" | "desc" };

type Props<T> = {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  initialSort?: Sort;
  /** Visually hidden caption naming the table. */
  caption: string;
  /** Optional detail row, opened by a toggle in the first cell. */
  expand?: (row: T) => ReactNode;
};

const MISSING = "—";

function compare(a: number | string | null, b: number | string | null, dir: "asc" | "desc"): number {
  // Missing values always sort last, whichever way the column is sorted.
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  const order = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
  return dir === "asc" ? order : -order;
}

/** The family data table: sortable headers, missing values last, optional row detail, scrolls in its own box. */
export function StatTable<T>({ rows, columns, rowKey, initialSort, caption, expand }: Props<T>) {
  const [sort, setSort] = useState<Sort | null>(initialSort ?? null);
  const [open, setOpen] = useState<string | null>(null);

  const sorted = sort
    ? [...rows].sort((a, b) => {
        const col = columns.find((c) => c.key === sort.key);
        return col ? compare(col.value(a), col.value(b), sort.dir) : 0;
      })
    : rows;

  const toggleSort = (col: Column<T>) =>
    setSort((cur) =>
      cur?.key === col.key
        ? { key: col.key, dir: cur.dir === "asc" ? "desc" : "asc" }
        : { key: col.key, dir: col.firstDir ?? (col.numeric ? "desc" : "asc") },
    );

  return (
    // relative: the visually hidden caption and tooltips are absolutely
    // positioned, and must stay inside this scroller or they widen the page.
    <div className="relative overflow-x-auto rounded-pr border border-pr-rule">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-pr-rule bg-pr-panel text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            {columns.map((col) => {
              const active = sort?.key === col.key;
              return (
                <th
                  key={col.key}
                  scope="col"
                  aria-sort={active ? (sort!.dir === "asc" ? "ascending" : "descending") : "none"}
                  className={`whitespace-nowrap px-3 py-2 font-semibold ${col.numeric ? "text-right" : ""}`}
                >
                  <button
                    type="button"
                    title={col.tooltip}
                    onClick={() => toggleSort(col)}
                    className={`inline-flex items-center gap-1 uppercase tracking-wide hover:text-pr-text ${active ? "text-pr-text" : ""}`}
                  >
                    {col.label}
                    <span aria-hidden="true">{active ? (sort!.dir === "asc" ? "↑" : "↓") : ""}</span>
                    {col.tooltip && <span className="sr-only">. {col.tooltip}</span>}
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const key = rowKey(row);
            const isOpen = open === key;
            return (
              <Fragment key={key}>
                <tr className="border-b border-pr-rule last:border-0">
                  {columns.map((col, i) => {
                    const v = col.value(row);
                    const content = col.render ? col.render(row) : v === null ? MISSING : v;
                    return (
                      <td key={col.key} className={`whitespace-nowrap px-3 py-2 ${col.numeric ? "text-right tabular-nums" : ""}`}>
                        {i === 0 && expand ? (
                          <button
                            type="button"
                            aria-expanded={isOpen}
                            aria-label={`${isOpen ? "Hide" : "Show"} ${String(v ?? key)} details`}
                            onClick={() => setOpen(isOpen ? null : key)}
                            className="inline-flex items-center gap-2 text-left font-semibold text-pr-text"
                          >
                            <span aria-hidden="true" className="text-pr-text-dim">{isOpen ? "▾" : "▸"}</span>
                            {content}
                          </button>
                        ) : (
                          content
                        )}
                      </td>
                    );
                  })}
                </tr>
                {expand && isOpen && (
                  <tr className="border-b border-pr-rule bg-pr-panel-2/40">
                    <td colSpan={columns.length} className="px-3 py-3">
                      {/* Pinned to the visible part of a wide table on a phone. */}
                      <div className="sticky left-3 max-w-[calc(100vw-4rem)]">{expand(row)}</div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
