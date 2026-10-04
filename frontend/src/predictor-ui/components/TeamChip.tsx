// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import { readableChip } from "../contrast";

// Team identity without logos: the team colour with its code set in ink
// that stays readable on that colour.
export function TeamChip({ code, name, color }: { code: string; name?: string; color?: string }) {
  const chip = color ? readableChip(color) : null;
  const style = chip ? { backgroundColor: chip.fill, color: chip.ink } : undefined;
  return (
    <span
      role={name ? "img" : undefined}
      aria-label={name}
      style={style}
      className={`inline-flex min-w-[3ch] ring-1 ring-inset ring-pr-rule items-center justify-center rounded-pr px-1.5 py-0.5 font-pr-display text-sm font-bold uppercase tracking-wide ${
        chip ? "" : "bg-pr-panel-2 text-pr-text"
      }`}
    >
      {code}
    </span>
  );
}
