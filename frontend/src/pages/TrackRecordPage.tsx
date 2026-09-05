import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TrackRecord } from "../types";

export function TrackRecordPage() {
  const [record, setRecord] = useState<TrackRecord | null>(null);

  useEffect(() => {
    api.trackRecord().then(setRecord);
  }, []);

  if (!record) {
    return <p>Loading…</p>;
  }

  return (
    <div>
      <h1>Track Record</h1>
      <p>Resolved games: {record.n_resolved_games}</p>
      <p>
        Moneyline accuracy:{" "}
        {record.pct_moneyline_correct != null ? `${Math.round(record.pct_moneyline_correct * 100)}%` : "No resolved games yet"}
      </p>
    </div>
  );
}
