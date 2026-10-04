// Synced from predictor-ui@490df2925aaf. Do not edit here: change predictor-hub/packages/predictor-ui and re-run scripts/sync-ui.mjs.
import type { ReactNode } from "react";

export type Sport = "pl" | "f1" | "nfl" | "cfb" | "nba" | "hub";
export type SiteLink = { sport: Sport; label: string; href: string };
export type Tab = { id: string; label: string };

type Props = {
  sport: Sport;
  /** "NBA", "F1", "NFL + CFB": shown as "<name> Predictor". */
  sportName: string;
  sites: SiteLink[];
  tabs: Tab[];
  activeTab: string;
  onTab: (id: string) => void;
  children: ReactNode;
};

/** The shared frame: family wordmark, a switcher to every sport, page tabs. */
export function AppFrame({ sport, sportName, sites, tabs, activeTab, onTab, children }: Props) {
  return (
    <div data-sport={sport} className="min-h-screen">
      <header className="border-b border-pr-rule">
        <div className="mx-auto flex max-w-screen-xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 pt-3 sm:px-6">
          <span className="font-pr-display text-sm font-bold uppercase tracking-[0.2em] text-pr-text-dim">Predictor</span>
          <nav aria-label="Sports" className="-mx-1 flex gap-1 overflow-x-auto">
            {sites.map((site) => {
              const current = site.sport === sport;
              return (
                <a
                  key={site.label}
                  href={site.href}
                  aria-current={current ? "page" : undefined}
                  className={`rounded-pr px-2.5 py-1 font-pr-display text-sm font-semibold uppercase tracking-wide transition-colors ${
                    current ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-dim hover:text-pr-text"
                  }`}
                >
                  {site.label}
                </a>
              );
            })}
          </nav>
        </div>
        <div className="mx-auto flex max-w-screen-xl flex-wrap items-end justify-between gap-x-6 gap-y-3 px-4 pb-0 pt-3 sm:px-6">
          <h1 className="pb-3 font-pr-display text-3xl font-bold uppercase tracking-wide text-pr-text">{sportName} Predictor</h1>
          <nav aria-label="Pages" className="flex gap-4 overflow-x-auto">
            {tabs.map((tab) => {
              const current = tab.id === activeTab;
              return (
                <button
                  key={tab.id}
                  type="button"
                  aria-current={current ? "page" : undefined}
                  onClick={() => onTab(tab.id)}
                  className={`whitespace-nowrap border-b-2 pb-3 text-sm font-semibold transition-colors ${
                    current ? "border-pr-accent text-pr-text" : "border-transparent text-pr-text-dim hover:text-pr-text"
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-screen-xl px-4 py-6 sm:px-6">{children}</main>
    </div>
  );
}
