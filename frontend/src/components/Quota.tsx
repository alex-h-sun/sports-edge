import type { Quota as QuotaT } from "../api";

export function Quota({ data }: { data: { quota: QuotaT | null } | null }) {
  const q = data?.quota;
  return (
    <div className="card">
      <h2>📊 Odds API Quota</h2>
      {q ? (
        <div className="tiles">
          <div className="tile">
            <span className="tile-label">Remaining</span>
            <span className="tile-value">{q.remaining}</span>
          </div>
          <div className="tile">
            <span className="tile-label">Used</span>
            <span className="tile-value">{q.used}</span>
          </div>
          <div className="tile">
            <span className="tile-label">Checked</span>
            <span className="tile-value small">
              {new Date(q.fetched_at).toLocaleString()}
            </span>
          </div>
        </div>
      ) : (
        <div className="muted">No quota data yet.</div>
      )}
    </div>
  );
}
