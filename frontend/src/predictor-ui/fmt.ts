// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
// The one owner of numbers on screen. Models and APIs keep full precision;
// rounding, signs, team names and time zones happen here, once, so every
// Predictor site reads the same.

const MINUS = "−";
const DASH = "—";

const bad = (x: number) => typeof x !== "number" || !Number.isFinite(x);

/** Whole percent. A live probability never reads 0% or 100%. */
export function pct(p: number): string {
  if (bad(p)) return DASH;
  if (p <= 0.005) return "<1%";
  if (p >= 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}

/** One decimal under 10% (where the difference matters), whole above. */
export function pctFine(p: number): string {
  if (bad(p)) return DASH;
  if (p < 0.001) return "<0.1%";
  // 0.0995 rounds to "10.0%" at one decimal; let whole percents take it.
  if (p < 0.1 && +(p * 100).toFixed(1) < 10) return `${(p * 100).toFixed(1)}%`;
  return pct(p);
}

/** One decimal: a projection of 16.514 carries no more meaning than 16.5. */
export function stat(x: number): string {
  if (bad(x)) return DASH;
  const r = +x.toFixed(1);
  return r < 0 ? `${MINUS}${Math.abs(r).toFixed(1)}` : r.toFixed(1);
}

/** Always signed, with a true minus sign. */
export function signed(x: number): string {
  if (bad(x)) return DASH;
  // Round first so 0.04 and -0.04 read as "0.0", not "+0.0" / "−0.0".
  const r = +x.toFixed(1);
  if (r > 0) return `+${r.toFixed(1)}`;
  if (r < 0) return `${MINUS}${Math.abs(r).toFixed(1)}`;
  return "0.0";
}

/** W3 / L1 instead of 3 / -1. */
export function streak(n: number): string {
  if (bad(n) || n === 0 || !Number.isInteger(n)) return DASH;
  return `${n > 0 ? "W" : "L"}${Math.abs(n)}`;
}

/** Hits over settled picks; nothing settled reads as a dash, never 0/0. */
export function record(hits: number, settled: number): string {
  return settled > 0 && !bad(hits) && !bad(settled) ? `${hits}/${settled}` : DASH;
}

/**
 * "MIA by 4.8". `x` is the home team's projected margin (home minus away),
 * so its sign picks the favourite; under half a point is a toss-up.
 */
export function margin(home: string, away: string, x: number): string {
  if (bad(x) || Math.abs(x) < 0.5) return "Toss-up";
  return `${x > 0 ? home : away} by ${Math.abs(x).toFixed(1)}`;
}

/** A betting line written with its team: "KC −3.5", "KC +3", "KC PK". */
export function spread(team: string, line: number): string {
  if (bad(line)) return DASH;
  if (line === 0) return `${team} PK`;
  const n = Number.isInteger(line) ? String(Math.abs(line)) : Math.abs(line).toFixed(1);
  return `${team} ${line < 0 ? MINUS : "+"}${n}`;
}

/**
 * A kickoff timestamp as a Date. An ISO date-time with no zone is UTC: the
 * NFL API sends "2026-10-04T17:00:00" meaning 17:00 UTC, which browsers
 * would otherwise read as local time (hours off, sometimes the wrong day).
 */
export function parseKickoff(iso: string): Date {
  const zoneless = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(iso);
  return new Date(zoneless ? `${iso}Z` : iso);
}

function formatParts(d: Date, timeZone: string | undefined, withTime: boolean): Record<string, string> {
  const options: Intl.DateTimeFormatOptions = { timeZone, weekday: "short", day: "numeric", month: "short" };
  if (withTime) Object.assign(options, { hour: "numeric", minute: "2-digit", timeZoneName: "short" });
  return Object.fromEntries(new Intl.DateTimeFormat("en-US", options).formatToParts(d).map((p) => [p.type, p.value]));
}

/** "Sat 3 Oct · 7:30 PM CDT": always shows the zone the time is in. A bare
 *  date ("2026-10-04") has no time, so it reads as the date alone. */
export function kickoff(iso: string, timeZone?: string): string {
  const dateOnly = /^\d{4}-\d{2}-\d{2}$/.test(iso);
  const d = dateOnly ? new Date(`${iso}T12:00:00Z`) : parseKickoff(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  let parts: Record<string, string>;
  try {
    parts = formatParts(d, dateOnly ? "UTC" : timeZone, !dateOnly);
  } catch {
    // An unknown zone name must not take the page down; use the viewer's.
    parts = formatParts(d, undefined, !dateOnly);
  }
  const day = `${parts.weekday} ${parts.day} ${parts.month}`;
  return dateOnly ? day : `${day} · ${parts.hour}:${parts.minute} ${parts.dayPeriod} ${parts.timeZoneName}`;
}

/** "v20260918120240" or an ISO timestamp → "Sep 18 model". */
export function modelDate(id: string): string {
  const m = /^v?(\d{4})-?(\d{2})-?(\d{2})/.exec(id);
  if (!m) return "Model";
  const [y, mo, day] = [+m[1], +m[2], +m[3]];
  const d = new Date(Date.UTC(y, mo - 1, day));
  // Reject impossible dates rather than letting Date roll "13/99" forward.
  if (d.getUTCFullYear() !== y || d.getUTCMonth() !== mo - 1 || d.getUTCDate() !== day) return "Model";
  return `${d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })} model`;
}
