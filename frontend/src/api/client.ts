import type {
  GameFacts,
  GamePrediction,
  GameSummary,
  PlayerPropPrediction,
  RetrainResponse,
  TrackRecord,
} from "../types";
import type { Signal } from "../predictor-ui";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

/** A GET that is NOT under `/api`.
 *
 * `/facts/{game_id}` is the one route this API serves at the root: the explainer
 * service is its caller and it addresses the API root, so mounting it under
 * `/api` would break that contract. `vite.config.ts` forwards `/facts` to the
 * backend for the same reason — otherwise the modal would work in production
 * (FastAPI serves the SPA and the route from one origin) and 404 in development,
 * which is the worst shape a difference can take. Everything else goes through
 * `get`, and the `/api/api/...` mistake this avoids is why there are two
 * functions rather than a flag.
 */
async function getAtRoot<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

/** What `GET /api/signals/{game_id}` answers. `signals` is empty rather than
 *  absent when the game has no honest signal — spec §2, "no data, no row". */
export interface SignalsResponse {
  signals: Signal[];
  sport?: string;
  id?: string;
}

export const api = {
  games: (season: number, week: number) => get<GameSummary[]>(`/games?season=${season}&week=${week}`),
  gamePrediction: (season: number, week: number, gameId: string) =>
    get<GamePrediction>(`/games/${season}/${week}/${gameId}/prediction`),
  playerProps: (season: number, week: number) => get<PlayerPropPrediction[]>(`/players/${season}/${week}/props`),
  trackRecord: () => get<TrackRecord>("/track-record"),
  retrain: () => post<RetrainResponse>("/retrain"),
  facts: (gameId: string) => getAtRoot<GameFacts>(`/facts/${encodeURIComponent(gameId)}`),
  signals: (gameId: string) =>
    get<SignalsResponse>(`/signals/${encodeURIComponent(gameId)}`),
};
