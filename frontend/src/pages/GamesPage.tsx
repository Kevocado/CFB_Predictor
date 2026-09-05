import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GamePrediction, GameSummary } from "../types";

export function GamesPage() {
  const [season, setSeason] = useState(2026);
  const [week, setWeek] = useState(1);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [predictions, setPredictions] = useState<Record<string, GamePrediction>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .games(season, week)
      .then(async (fetchedGames) => {
        setGames(fetchedGames);
        const entries = await Promise.all(
          fetchedGames.map(async (g) => {
            try {
              const prediction = await api.gamePrediction(season, week, g.game_id);
              return [g.game_id, prediction] as const;
            } catch {
              return null;
            }
          }),
        );
        setPredictions(Object.fromEntries(entries.filter((e): e is [string, GamePrediction] => e !== null)));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [season, week]);

  return (
    <div>
      <h1>Week {week} Games</h1>
      <div>
        <label>
          Season:{" "}
          <input type="number" value={season} onChange={(e) => setSeason(Number(e.target.value))} />
        </label>
        <label>
          Week:{" "}
          <input type="number" min={1} max={20} value={week} onChange={(e) => setWeek(Number(e.target.value))} />
        </label>
      </div>
      {loading && <p>Loading…</p>}
      {error && <p role="alert">{error}</p>}
      <ul>
        {games.map((game) => {
          const prediction = predictions[game.game_id];
          return (
            <li key={game.game_id}>
              <strong>{game.away_team} @ {game.home_team}</strong> — {new Date(game.gameday).toLocaleDateString()}
              {prediction && (
                <span>
                  {" "}— Home win {Math.round(prediction.home_win_prob * 100)}%
                  {prediction.over_prob != null && ` · Over ${Math.round(prediction.over_prob * 100)}%`}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
