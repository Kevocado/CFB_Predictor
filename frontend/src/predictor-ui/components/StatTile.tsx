// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";

export function StatTile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-pr border border-pr-rule bg-pr-panel p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim">{label}</div>
      <div className="mt-1 font-pr-display text-3xl font-semibold text-pr-text">{value}</div>
      {sub && <div className="mt-1 text-xs text-pr-text-faint">{sub}</div>}
    </div>
  );
}
