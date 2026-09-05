import type {
  GamePrediction,
  GameSummary,
  PlayerPropPrediction,
  RetrainResponse,
  TrackRecord,
} from "../types";

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

export const api = {
  games: (season: number, week: number) => get<GameSummary[]>(`/games?season=${season}&week=${week}`),
  gamePrediction: (season: number, week: number, gameId: string) =>
    get<GamePrediction>(`/games/${season}/${week}/${gameId}/prediction`),
  playerProps: (season: number, week: number) => get<PlayerPropPrediction[]>(`/players/${season}/${week}/props`),
  trackRecord: () => get<TrackRecord>("/track-record"),
  retrain: () => post<RetrainResponse>("/retrain"),
};
