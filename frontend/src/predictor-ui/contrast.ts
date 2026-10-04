// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
// WCAG 2.x relative luminance and contrast ratio, for the token gate and for
// picking readable ink on arbitrary team colours at runtime.

const DARK_INK = "#0b0d10";
const LIGHT_INK = "#f3f5f8";

/** "#fff" → "#ffffff", "#E31837" → "#e31837"; anything else → null. */
export function parseHex(value: string): string | null {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value.trim());
  if (!m) return null;
  const hex = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1];
  return `#${hex.toLowerCase()}`;
}

function channel(c: number): number {
  const s = c / 255;
  return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}

function rgb(hex: string): [number, number, number] {
  const n = parseInt((parseHex(hex) ?? "#000000").slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function luminance(hex: string): number {
  const [r, g, b] = rgb(hex);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** Dark or light ink, whichever reads better on `background`. */
export function inkFor(background: string, dark = DARK_INK, light = LIGHT_INK): string {
  return contrast(dark, background) >= contrast(light, background) ? dark : light;
}

function mix(hex: string, toward: [number, number, number], amount: number): string {
  const out = rgb(hex).map((c, i) => Math.round(c + (toward[i] - c) * amount));
  return `#${out.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/**
 * A team-colour chip whose code stays readable (4.5:1, the bar for 14px
 * text). The team's own colour is kept when it already reads; mid-tones like
 * Chiefs red are nudged darker (or lighter) until one ink clears the bar.
 * Returns null for values that are not hex, so callers fall back to neutral.
 */
export function readableChip(colour: string): { fill: string; ink: string } | null {
  const fill = parseHex(colour);
  if (!fill) return null;
  for (let step = 0; step <= 10; step += 1) {
    for (const [ink, toward] of [
      [LIGHT_INK, [0, 0, 0]],
      [DARK_INK, [255, 255, 255]],
    ] as const) {
      const candidate = step === 0 ? fill : mix(fill, [...toward], step / 10);
      if (contrast(ink, candidate) >= 4.5) return { fill: candidate, ink };
    }
  }
  return { fill: "#1d232c", ink: LIGHT_INK };
}
