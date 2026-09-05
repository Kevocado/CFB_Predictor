import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerPropPrediction } from "../types";

export function PlayerPropsPage({ season, week }: { season: number; week: number }) {
  const [props, setProps] = useState<PlayerPropPrediction[]>([]);
  const [sortBy, setSortBy] = useState<"anytime_td_prob" | "rushing_yards" | "receiving_yards" | "passing_yards">(
    "anytime_td_prob",
  );

  useEffect(() => {
    api.playerProps(season, week).then(setProps);
  }, [season, week]);

  const sorted = [...props].sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  return (
    <div>
      <h1>Player Props — Week {week}</h1>
      <label>
        Sort by:{" "}
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}>
          <option value="anytime_td_prob">Anytime TD</option>
          <option value="passing_yards">Passing Yards</option>
          <option value="rushing_yards">Rushing Yards</option>
          <option value="receiving_yards">Receiving Yards</option>
        </select>
      </label>
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>Anytime TD</th>
            <th>Passing Yds</th>
            <th>Rushing Yds</th>
            <th>Receiving Yds</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((player) => (
            <tr key={player.player_id}>
              <td>{player.player_name}</td>
              <td>{Math.round(player.anytime_td_prob * 100)}%</td>
              <td>{player.passing_yards != null ? Math.round(player.passing_yards) : "—"}</td>
              <td>{player.rushing_yards != null ? Math.round(player.rushing_yards) : "—"}</td>
              <td>{player.receiving_yards != null ? Math.round(player.receiving_yards) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
