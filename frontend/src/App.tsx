import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import type { Meta, EdgesResponse } from "./api";
import { SPORT_LABELS } from "./constants";
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
      reloadMeta={loadMeta}
      onLogout={async () => {
        // Tell the server to drop the session, then hard-reload so auth is
        // re-checked against the server (the source of truth) and all
        // in-memory state is cleared. Runs even if the request fails.
        try {
          await api.logout();
        } finally {
          window.location.reload();
        }
      }}
    />
  );
}

function Dashboard({
  meta,
  reloadMeta,
  onLogout,
}: {
  meta: Meta;
  reloadMeta: () => void;
  onLogout: () => void;
}) {
  const [sports, setSports] = useState<string[]>(meta.default_sports);
  const [markets, setMarkets] = useState<string[]>(["Moneyline", "Totals", "Props"]);
  const [minEdge, setMinEdge] = useState<number>(meta.min_edge);
  const [reloading, setReloading] = useState(false);
  const [pulling, setPulling] = useState(false);
  const [pullMsg, setPullMsg] = useState<string | null>(null);
  const [pullErr, setPullErr] = useState<string | null>(null);

  // Edges are computed only on explicit request: the pipeline builds features
  // and scores every market, which takes several seconds. Nothing runs until
  // the user presses "Find value bets".
  const [edges, setEdges] = useState<EdgesResponse | null>(null);
  const [edgesLoading, setEdgesLoading] = useState(false);
  const [edgesError, setEdgesError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  const query = { sports, minEdge, markets };

  const searchEdges = useCallback(() => {
    setSearched(true);
    setEdgesLoading(true);
    setEdgesError(null);
    api
      .edges({ sports, minEdge, markets })
      .then((d) => setEdges(d))
      .catch((e: unknown) => setEdgesError(e instanceof Error ? e.message : String(e)))
      .finally(() => setEdgesLoading(false));
  }, [sports, markets, minEdge]);

  // These are cheap reads, so they load on mount as before.
  const injuries = useAsync(() => api.injuries(), []);
  const quota = useAsync(() => api.quota(), []);
  const bankroll = useAsync(() => api.bankroll(), []);

  async function reloadSnapshot() {
    setReloading(true);
    try {
      await api.reload();
      if (searched) searchEdges(); // only re-run edges if the user already searched
      injuries.reload();
      quota.reload();
      bankroll.reload();
    } catch {
      // surfaced as a no-op; the panels keep their last data
    } finally {
      setReloading(false);
    }
  }

  // Live pull: ingest fresh games/odds and recompute edges against the shared DB —
  // the same work `python run.py` does, so the terminal and the web app stay in sync.
  async function pullData() {
    const labels = sports.map((s) => SPORT_LABELS[s] ?? s).join(", ");
    if (
      !window.confirm(
        `Pull fresh games + odds for ${labels} now?\n` +
          "This fetches live odds and uses your Odds API quota."
      )
    ) {
      return;
    }
    setPulling(true);
    setPullErr(null);
    setPullMsg(null);
    try {
      const r = await api.pull({ sports });
      const parts = [`${r.n_edges} edge${r.n_edges === 1 ? "" : "s"}`, `${r.duration_s}s`];
      if (r.quota) parts.push(`quota ${r.quota.remaining} left`);
      const problems = [
        ...r.stages
          .filter((s) => s.status === "error")
          .map((s) => [s.sport, s.stage].filter(Boolean).join(" ")),
        ...r.edge_errors,
      ];
      if (problems.length) parts.push(`issues: ${problems.join("; ")}`);
      setPullMsg(`Pulled ${r.sports.join(", ")} · ${parts.join(" · ")}`);
      reloadMeta(); // refresh the header's snapshot timestamp
      if (searched) searchEdges();
      injuries.reload();
      quota.reload();
      bankroll.reload();
    } catch (e) {
      setPullErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPulling(false);
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
          <button
            className="btn"
            onClick={pullData}
            disabled={pulling || reloading}
            title="Ingest fresh games + odds and recompute edges (uses Odds API quota)"
          >
            {pulling ? "Pulling…" : "⤓ Pull data"}
          </button>
          <button
            className="btn ghost"
            onClick={reloadSnapshot}
            disabled={reloading || pulling}
            title="Re-download the published snapshot from storage (read-only)"
          >
            {reloading ? "Reloading…" : "↻ Reload snapshot"}
          </button>
          <button className="btn ghost" onClick={onLogout}>Sign out</button>
        </div>
      </header>

      <main>
        {(pullErr || pullMsg) && (
          <div className={`card pull-status${pullErr ? " error" : ""}`}>
            {pullErr ? `Pull failed: ${pullErr}` : pullMsg}
          </div>
        )}
        <Controls
          sports={sports}
          setSports={setSports}
          markets={markets}
          setMarkets={setMarkets}
          minEdge={minEdge}
          setMinEdge={setMinEdge}
          allSports={meta.sports}
        />
        {searched && <SummaryTiles data={edges} />}
        <ValueBets
          data={edges}
          loading={edgesLoading}
          error={edgesError}
          searched={searched}
          onSearch={searchEdges}
          query={query}
        />
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
