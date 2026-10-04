// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
/** SummaryButton — the only network caller in the panel.
 *
 *  One job: turn a press into a summary or an unavailable state, and nothing
 *  else. The request function is injected, so the button carries no fetch, no
 *  URL and no knowledge of the route — a test stubs the request and the network
 *  is never touched.
 *
 *  Three failure modes, one outcome. A rejecting request (a dead container, a
 *  proxy 502), a request that outruns the budget, and a 2xx whose body is not
 *  the expected shape all funnel to `onUnavailable`. The fixed upstream message
 *  is never shown to a reader: the button offers a retry and the flow is
 *  untouched.
 *
 *  A second press cannot double-fire: the button is disabled while a request is
 *  in flight, and the handler guards on the same flag.
 */
import { useEffect, useRef, useState } from "react";
import type { Explanation } from "./ExplainerPanel";

/** The 15 s budget from the spec's error table. */
const DEFAULT_BUDGET_MS = 15_000;

/** Whether a resolved value is the v2 shape the panel can render. A 2xx that
 *  is not this shape is treated as unavailable, never rendered. */
function isSummary(value: unknown): value is Explanation {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof (value as { verdict?: unknown }).verdict === "string" &&
    Array.isArray((value as { factors?: unknown }).factors)
  );
}

export interface SummaryButtonProps {
  /** The injected request function. The only network caller. */
  request: () => Promise<unknown>;
  /** Called when a valid summary arrives. */
  onSummary: (summary: Explanation) => void;
  /** Called when the request fails, times out, or returns an unexpected shape. */
  onUnavailable: () => void;
  /** The button's label. The parent drives this from its state, so a retry
   *  reads as a retry. */
  label?: string;
  /** The budget in milliseconds. Defaults to the spec's 15 s; a test injects a
   *  smaller value to exercise the timeout without waiting. */
  budgetMs?: number;
}

export function SummaryButton({
  request,
  onSummary,
  onUnavailable,
  label = "Get the AI summary",
  budgetMs = DEFAULT_BUDGET_MS,
}: SummaryButtonProps) {
  const [loading, setLoading] = useState(false);
  // Held in a ref so an unmount mid-request clears the timer rather than
  // rejecting into a component that is gone.
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Whether this button is still on the page. The timer was cleared on unmount
  // and the comment above said that was enough; it is not. A cleared timer
  // stops the BUDGET rejecting, but `request()`'s own promise is still pending
  // and still resolves — calling `onSummary`/`onUnavailable`/`setLoading` on a
  // component that no longer exists, which in React 19 is a warning at best and
  // a state update on an unmounted tree at worst. The reader's race is real:
  // they press the button and close the dialog.
  const mounted = useRef(true);

  useEffect(() => {
    // Restored in setup, not just cleared in cleanup: StrictMode mounts,
    // unmounts, and remounts, so the cleanup runs while the component is still
    // here. Without this line the remount inherits mounted.current = false and
    // every guard treats the live button as gone — the first click resolves
    // into nothing and the button stays disabled.
    mounted.current = true;
    return () => {
      mounted.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const handleClick = () => {
    if (loading) return; // a second press cannot double-fire
    setLoading(true);
    const budget = new Promise<never>((_, reject) => {
      timer.current = setTimeout(() => reject(new Error("The summary took too long.")), budgetMs);
    });
    // `Promise.resolve().then(request)` rather than `request()` bare: a request
    // that throws SYNCHRONOUSLY (a bad URL, a missing token read at call time)
    // used to escape before the race was ever built, so no `.catch` saw it, the
    // button stayed disabled forever, and the page showed a dead control. One
    // tick of deferral moves the throw inside the chain.
    Promise.race([Promise.resolve().then(request), budget])
      .then((value) => {
        if (!mounted.current) return;
        if (isSummary(value)) onSummary(value);
        else onUnavailable();
      })
      .catch(() => {
        // A proxy 502, a dead container, a timeout — all unavailable. The fixed
        // upstream message is never shown to a reader.
        if (!mounted.current) return;
        onUnavailable();
      })
      .finally(() => {
        if (timer.current) clearTimeout(timer.current);
        timer.current = null;
        if (mounted.current) setLoading(false);
      });
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={loading}
      aria-busy={loading}
      className="self-start rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text transition-colors hover:border-pr-accent disabled:opacity-60"
    >
      {loading ? "Writing…" : label}
    </button>
  );
}
