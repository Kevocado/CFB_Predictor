import { useState } from "react";
import { GamesPage } from "./pages/GamesPage";
import { PlayerPropsPage } from "./pages/PlayerPropsPage";
import { TrackRecordPage } from "./pages/TrackRecordPage";

type Tab = "games" | "props" | "track-record";

function App() {
  const [tab, setTab] = useState<Tab>("games");
  const season = 2026;
  const week = 1;

  return (
    <div>
      <nav>
        <button onClick={() => setTab("games")}>Games</button>
        <button onClick={() => setTab("props")}>Player Props</button>
        <button onClick={() => setTab("track-record")}>Track Record</button>
      </nav>
      {tab === "games" && <GamesPage />}
      {tab === "props" && <PlayerPropsPage season={season} week={week} />}
      {tab === "track-record" && <TrackRecordPage />}
    </div>
  );
}

export default App;
