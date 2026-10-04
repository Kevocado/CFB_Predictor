// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
const actionClass =
  "rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text transition-colors hover:border-pr-accent";

/** Named loading state: says what is loading, and is announced. */
export function Skeleton({ label }: { label: string }) {
  return (
    <div role="status" aria-live="polite" className="flex flex-col gap-2 py-4">
      <span className="text-sm text-pr-text-dim">{label}</span>
      <div aria-hidden="true" className="h-2 w-full max-w-md rounded-pr bg-pr-panel-2 motion-safe:animate-pulse" />
    </div>
  );
}

/** Empty is never a dead end: it says why and offers the next step. */
export function EmptyState({ message, action }: { message: string; action?: { label: string; onClick: () => void } }) {
  return (
    <div role="status" aria-live="polite" className="flex flex-wrap items-center gap-3 py-4 text-sm text-pr-text-dim">
      <span>{message}</span>
      {action && (
        <button type="button" onClick={action.onClick} className={actionClass}>
          {action.label}
        </button>
      )}
    </div>
  );
}

/** Errors say what failed and how to recover; never a raw error string. */
export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
      <span>{message}</span>
      <button type="button" onClick={onRetry} className={actionClass}>
        Try again
      </button>
    </div>
  );
}
