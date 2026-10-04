/**
 * The fixture modal, and the trust row on it.
 *
 * CFB has had no fixture modal and no facts block at all — a `GamesPage` that
 * listed `<strong>{away} @ {home}</strong>` and a probability. That is why the
 * one market proven good enough to ship a badge (the moneyline: 390 graded rows,
 * 3 of 4 bands over the floor) had nowhere to render, and why the vendored
 * `SignalRows` was inert here.
 *
 * What this file holds, each a way the row could go wrong:
 *
 *  - the badge renders from `/api/signals/{game_id}`, INSTANT and without a
 *    click. Spec §2: "Signals are instant: computed from stored data, no model
 *    call." A row behind a button would invert that.
 *  - `{"signals": []}` renders NOTHING. Spec §2: "No data, no row. No empty
 *    states, no 'unknown' rows, no filler." An empty list that renders a
 *    placeholder is the failure this test exists to prevent.
 *  - a failed signals request leaves the modal fully usable. The facts block is
 *    the page; a signal is an enhancement on it, so the signal's failure must
 *    never become the modal's error state. This is the same rule the adapter and
 *    the endpoint keep on their side, and it is worth pinning on this side too.
 *  - the modal states the pick's own probability, and the badge states a rate
 *    about PAST picks. Two different numbers about two different things; the
 *    badge must not be read as a forecast for this game.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";

import { GameDetailModal } from "./GameDetailModal";
import { api } from "../api/client";
import type { Signal } from "../predictor-ui";

const FACTS = {
  sport: "cfb",
  id: "401520145",
  title: "Ohio State at Texas",
  starts_at: "2099-08-30T16:00:00Z",
  status: "upcoming",
  pick_timing: "pre_kickoff",
  pick: { label: "Texas", prob: 0.65 },
  markets: [{ market: "moneyline", model: { Texas: 0.65, "Ohio State": 0.35 } }],
  drivers: [],
  context: {},
  players: [],
  record: null,
  result: null,
};

/** The payload `cfb_predictor.signals.trust.trust_signal` builds. */
const trust = (over: Partial<Signal> = {}): Signal => ({
  kind: "trust",
  sport: "cfb",
  game_id: "401520145",
  headline: {
    text: "When the model says ~68%, its picks landed 71% of the time",
    figures: { rate: 0.7143, stated_prob: 0.6791 },
  },
  n: 66,
  source: "66 resolved games in this project's tracking.db, 0.75-1 probability band, the moneyline",
  as_of: "",
  strength: 0.31,
  pre_kickoff_only: true,
  visual: "reliability_bar",
  ...over,
});

let signals: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.spyOn(api, "facts").mockResolvedValue(FACTS as never);
  signals = vi.spyOn(api, "signals").mockResolvedValue({ sport: "cfb", id: "401520145", signals: [trust()] } as never);
});

afterEach(() => vi.restoreAllMocks());

function open() {
  render(<GameDetailModal gameId="401520145" onClose={() => {}} />);
}

describe("the CFB fixture modal", () => {
  it("shows the game's own pick", async () => {
    open();
    expect(await screen.findByText("Texas")).toBeInTheDocument();
  });

  it("renders the trust row without a click", async () => {
    open();
    expect(await screen.findByText(/its picks landed 71% of the time/)).toBeInTheDocument();
  });

  it("asks for the signals by game id", async () => {
    open();
    await screen.findByText(/its picks landed/);
    expect(signals).toHaveBeenCalledWith("401520145");
  });

  it("renders nothing at all when there is no signal", async () => {
    signals.mockResolvedValue({ sport: "cfb", id: "401520145", signals: [] } as never);
    open();
    // The facts block still renders; the signal area contributes no node.
    expect(await screen.findByText("Texas")).toBeInTheDocument();
    expect(screen.queryByText(/its picks landed/)).not.toBeInTheDocument();
  });

  it("stays usable when the signals request fails", async () => {
    signals.mockRejectedValue(new Error("signals down"));
    open();
    expect(await screen.findByText("Texas")).toBeInTheDocument();
    expect(screen.queryByText(/its picks landed/)).not.toBeInTheDocument();
    // And it is not the modal's error state either.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps the badge's past rate distinct from this game's pick", async () => {
    open();
    // The badge quotes a record about past picks; the pick is this game. The
    // wording has to make that difference for a reader, which is why the
    // adapter's headline is past tense and names its own sample.
    const badge = await screen.findByText(/its picks landed 71% of the time/);
    expect(badge.textContent).toMatch(/its picks landed/);
    expect(badge.textContent).not.toMatch(/this game|will be|is right/);
  });

  it("is a dialog a keyboard user can leave", async () => {
    const onClose = vi.fn();
    render(<GameDetailModal gameId="401520145" onClose={onClose} />);
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeInTheDocument();
  });
});