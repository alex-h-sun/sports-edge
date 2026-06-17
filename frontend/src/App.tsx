import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import type { Meta } from "./api";
import { useAsync } from "./hooks";
import { Login } from "./components/Login";
import { Controls } from "./components/Controls";
import { SummaryTiles } from "./components/SummaryTiles";
import { ValueBets } from "./components/ValueBets";
import { Injuries } from "./components/Injuries";
import { Quota } from "./components/Quota";
import { Bankroll } from "./components/Bankroll";
import { ManualCalculator } from "./components/ManualCalculator";

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [authed, setAuthed] = useState<boolean | null>(null);

  const loadMeta = useCallback(() => {
    api
      .meta()
      .then((m) => {
        setMeta(m);
        setAuthed(true);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) setAuthed(false);
        else setAuthed(false);
      });
  }, []);

  useEffect(loadMeta, [loadMeta]);

  if (authed === null) return <div className="loading">Loading…</div>;
  if (!authed || !meta) return <Login onSuccess={loadMeta} />;

  return (
    <Dashboard
      meta={meta}
      onLogout={() => {
        void api.logout().finally(() => setAuthed(false));
      }}
    />
  );
}

function Dashboard({ meta, onLogout }: { meta: Meta; onLogout: () => void }) {
  const [sports, setSports] = useState<string[]>(meta.default_sports);
  const [markets, setMarkets] = useState<string[]>(["Moneyline", "Totals", "Props"]);
  const [minEdge, setMinEdge] = useState<number>(meta.min_edge);
  const [reloading, setReloading] = useState(false);

  const query = { sports, minEdge, markets };
  const edges = useAsync(() => api.edges(query), [sports.join(","), markets.join(","), minEdge]);
  const injuries = useAsync(() => api.injuries(), []);
  const quota = useAsync(() => api.quota(), []);
  const bankroll = useAsync(() => api.bankroll(), []);

  async function reloadSnapshot() {
    setReloading(true);
    try {
      await api.reload();
      edges.reload();
      injuries.reload();
      quota.reload();
      bankroll.reload();
    } catch {
      // surfaced as a no-op; the panels keep their last data
    } finally {
      setReloading(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">🎯 Sports Edge</div>
        <div className="meta">
          <span className="muted">{meta.db_label}</span>
          {meta.snapshot_ts && (
            <span className="muted">· snapshot {new Date(meta.snapshot_ts).toLocaleString()}</span>
          )}
          <button className="btn ghost" onClick={reloadSnapshot} disabled={reloading}>
            {reloading ? "Reloading…" : "↻ Reload snapshot"}
          </button>
          <button className="btn ghost" onClick={onLogout}>Sign out</button>
        </div>
      </header>

      <main>
        <Controls
          sports={sports}
          setSports={setSports}
          markets={markets}
          setMarkets={setMarkets}
          minEdge={minEdge}
          setMinEdge={setMinEdge}
          allSports={meta.sports}
        />
        <SummaryTiles data={edges.data} />
        <ValueBets data={edges.data} loading={edges.loading} query={query} />
        <ManualCalculator />
        <div className="two-up">
          <Injuries data={injuries.data} />
          <Quota data={quota.data} />
        </div>
        <Bankroll data={bankroll.data} />
      </main>
    </div>
  );
}
