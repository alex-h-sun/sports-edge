import type { EdgesResponse } from "../api";
import { money, pct } from "../format";

export function SummaryTiles({ data }: { data: EdgesResponse | null }) {
  const edges = data?.edges ?? [];
  const totalKelly = edges.reduce((s, e) => s + (e.kelly_stake || 0), 0);
  const best = edges.reduce((m, e) => Math.max(m, e.edge || 0), 0);

  return (
    <div className="tiles">
      <div className="tile">
        <span className="tile-label">Edges found</span>
        <span className="tile-value">{edges.length}</span>
      </div>
      <div className="tile">
        <span className="tile-label">Total Kelly</span>
        <span className="tile-value">{money(totalKelly)}</span>
      </div>
      <div className="tile">
        <span className="tile-label">Best edge</span>
        <span className="tile-value">{pct(best)}</span>
      </div>
      <div className="tile">
        <span className="tile-label">Min edge</span>
        <span className="tile-value">{pct(data?.min_edge ?? 0)}</span>
      </div>
    </div>
  );
}
