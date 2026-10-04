// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { record as fmtRecord } from "../fmt";
import type { Moment } from "./StatusBadge";

type RoundRecord = { hits: number; settled: number; rebuilt: number };

type Props = {
  label: string;
  /** "gameweek", "week", "round": used in "Jump to current …" and the record line. */
  unit?: string;
  canPrev: boolean;
  canNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  onJumpToCurrent?: () => void;
  record?: RoundRecord;
  moment?: Moment;
};

const stepClass =
  "rounded-pr border border-pr-rule bg-pr-panel px-2.5 py-1.5 text-pr-text-dim transition-colors hover:text-pr-text disabled:cursor-not-allowed disabled:opacity-40";

function recordLine(r: RoundRecord, unit: string, moment: Moment): string | null {
  if (r.settled > 0) return `${fmtRecord(r.hits, r.settled)} picks made before ${moment} correct`;
  if (r.rebuilt > 0) return `No picks made before ${moment} this ${unit}`;
  return null;
}

/** Round stepping plus the round's honest record of picks made before the start. */
export function RoundNavigator({ label, unit = "week", canPrev, canNext, onPrev, onNext, onJumpToCurrent, record, moment = "kickoff" }: Props) {
  const line = record ? recordLine(record, unit, moment) : null;
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" aria-label={`Previous ${unit}`} disabled={!canPrev} onClick={onPrev} className={stepClass}>
          ←
        </button>
        <h2 className="whitespace-nowrap font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">{label}</h2>
        <button type="button" aria-label={`Next ${unit}`} disabled={!canNext} onClick={onNext} className={stepClass}>
          →
        </button>
        {onJumpToCurrent && (
          <button type="button" onClick={onJumpToCurrent} className="whitespace-nowrap rounded-pr border border-pr-accent px-2.5 py-1.5 text-xs font-semibold text-pr-text transition-colors hover:bg-pr-panel-2">
            Jump to current {unit}
          </button>
        )}
      </div>
      {line && <p className="text-sm text-pr-text-dim">{line}</p>}
    </div>
  );
}
