import { api } from "../api";
import type { EdgeQuery, EdgesResponse } from "../api";
import { edgeDetail, money, odds, pct } from "../format";

interface Props {
  data: EdgesResponse | null;
  loading: boolean;
  error: string | null;
  searched: boolean;
  onSearch: () => void;
  query: EdgeQuery;
}

export function ValueBets({ data, loading, error, searched, onSearch, query }: Props) {
  const edges = data?.edges ?? [];
  const errors = data?.errors ?? [];

  return (
    <div className="card">
      <div className="card-head">
        <h2>📋 Value Bets</h2>
        <div className="row-actions">
          {searched && edges.length > 0 && (
            <a className="btn ghost" href={api.edgesCsvUrl(query)}>⬇ CSV</a>
          )}
          <button className="btn" onClick={onSearch} disabled={loading}>
            {loading ? "Searching…" : searched ? "↻ Refresh" : "Find value bets"}
          </button>
        </div>
      </div>

      {error && <div className="warn">{error}</div>}
      {errors.length > 0 && <div className="warn">{errors.join(" · ")}</div>}

      {!searched ? (
        <div className="muted">
          Press “Find value bets” to score the selected markets against the current
          odds. This builds features and runs every model, so it takes a few seconds.
        </div>
      ) : loading ? (
        <div className="muted">Searching…</div>
      ) : edges.length === 0 ? (
        <div className="muted">
          No edges above threshold. Live edges need fresh odds and upcoming games in
          the published snapshot.
        </div>
      ) : (
        <table className="grid">
          <thead>
            <tr>
              <th>Sport</th><th>Market</th><th>Game</th><th>Bet</th>
              <th>Odds</th><th>Edge</th><th>Kelly $</th><th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((e, i) => (
              <tr key={i}>
                <td>{e.sport.toUpperCase()}</td>
                <td>{e.market}</td>
                <td>{e.game}</td>
                <td>{e.bet}</td>
                <td>{odds(e.odds)}</td>
                <td className="pos">{pct(e.edge)}</td>
                <td>{money(e.kelly_stake)}</td>
                <td className="muted">{edgeDetail(e)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
