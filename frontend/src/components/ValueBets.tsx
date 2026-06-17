import { api } from "../api";
import type { EdgeQuery, EdgesResponse } from "../api";
import { edgeDetail, money, odds, pct } from "../format";

interface Props {
  data: EdgesResponse | null;
  loading: boolean;
  query: EdgeQuery;
}

export function ValueBets({ data, loading, query }: Props) {
  const edges = data?.edges ?? [];
  const errors = data?.errors ?? [];

  return (
    <div className="card">
      <div className="card-head">
        <h2>📋 Value Bets</h2>
        {edges.length > 0 && (
          <a className="btn" href={api.edgesCsvUrl(query)}>⬇ CSV</a>
        )}
      </div>

      {errors.length > 0 && <div className="warn">{errors.join(" · ")}</div>}

      {loading ? (
        <div className="muted">Loading…</div>
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
