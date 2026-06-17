import type { Bankroll as BankrollT } from "../api";
import { money, pct } from "../format";
import { EquityChart } from "./EquityChart";

const LEDGER_COLS = [
  "placed_date", "game_date", "sport", "market", "game",
  "selection", "odds", "edge", "stake", "status", "profit", "balance",
];

export function Bankroll({ data }: { data: BankrollT | null }) {
  if (!data) return null;
  const s = data.summary;

  return (
    <div className="card">
      <h2>💰 Bankroll Simulator</h2>
      {!s ? (
        <div className="muted">
          No paper-trading bets yet. Run <code>python run.py</code> on a day with
          moneyline edges to start the ledger.
        </div>
      ) : (
        <>
          <div className="tiles">
            <div className="tile">
              <span className="tile-label">Balance</span>
              <span className="tile-value">{money(s.current)}</span>
              <span className={s.total_return_pct >= 0 ? "pos" : "neg"}>
                {s.total_return_pct >= 0 ? "+" : ""}
                {s.total_return_pct.toFixed(2)}%
              </span>
            </div>
            <div className="tile">
              <span className="tile-label">Hit rate</span>
              <span className="tile-value">{pct(s.hit_rate)}</span>
              <span className="muted">{s.n_settled} settled</span>
            </div>
            <div className="tile">
              <span className="tile-label">Max drawdown</span>
              <span className="tile-value">{s.max_drawdown_pct.toFixed(1)}%</span>
            </div>
            <div className="tile">
              <span className="tile-label">Open / void</span>
              <span className="tile-value">{s.n_open} / {s.n_void}</span>
            </div>
          </div>

          <EquityChart points={data.curve} />

          <details>
            <summary>Ledger ({data.ledger.length} bets)</summary>
            <div className="scroll">
              <table className="grid small">
                <thead>
                  <tr>{LEDGER_COLS.map((c) => <th key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {data.ledger.map((r, i) => (
                    <tr key={i}>
                      {LEDGER_COLS.map((c) => <td key={c}>{String(r[c] ?? "")}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </div>
  );
}
