import { useEffect, useState } from "react";
import { api } from "../api";
import type { Edge, ManualOptions } from "../api";
import { edgeDetail, money, odds, pct } from "../format";

const SPORTS = [
  { key: "tennis", label: "Tennis" },
  { key: "nba", label: "NBA" },
  { key: "nhl", label: "NHL" },
];

export function ManualCalculator() {
  const [sport, setSport] = useState("nba");
  const [market, setMarket] = useState("moneyline");
  const [opts, setOpts] = useState<ManualOptions | null>(null);

  const [sideA, setSideA] = useState("");
  const [sideB, setSideB] = useState("");
  const [player, setPlayer] = useState("");
  const [stat, setStat] = useState("");
  const [surface, setSurface] = useState("");
  const [line, setLine] = useState(220.5);
  const [oddsA, setOddsA] = useState(-150);
  const [oddsB, setOddsB] = useState(130);
  const [overOdds, setOverOdds] = useState(-110);
  const [underOdds, setUnderOdds] = useState(-110);

  const [result, setResult] = useState<Edge[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const markets = ["moneyline", "totals", "spread", ...(sport === "nba" ? ["prop"] : [])];
  const isTennis = sport === "tennis";

  useEffect(() => {
    setOpts(null);
    setResult(null);
    setError(null);
    if (!markets.includes(market)) setMarket("moneyline");
    api.manualOptions(sport).then(setOpts).catch((e) =>
      setError(e instanceof Error ? e.message : String(e)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sport]);

  useEffect(() => {
    if (!opts) return;
    setSideA(opts.entities[0] ?? "");
    setSideB(opts.entities[1] ?? opts.entities[0] ?? "");
    setPlayer(opts.players[0] ?? "");
    setStat(opts.prop_stats[0] ?? "");
  }, [opts]);

  const entities = opts?.entities ?? [];
  const players = opts?.players ?? [];
  const propStats = opts?.prop_stats ?? [];

  async function calc() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const body: Record<string, unknown> = { sport, market };
      if (market === "prop") {
        body.player = player;
        body.stat = stat;
        body.line = line;
        body.over_odds = overOdds;
        body.under_odds = underOdds;
      } else {
        body.side_a = sideA;
        body.side_b = sideB;
        if (isTennis && surface) body.surface = surface;
        if (market === "totals") {
          body.line = line;
          body.over_odds = overOdds;
          body.under_odds = underOdds;
        } else if (market === "spread") {
          body.line = line;
          body.odds_a = oddsA;
          body.odds_b = oddsB;
        } else {
          body.odds_a = oddsA;
          body.odds_b = oddsB;
        }
      }
      const res = await api.manualEdge(body);
      setResult(res.edges);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const best = result && result.length
    ? result.reduce((a, b) => (b.edge > a.edge ? b : a))
    : null;

  return (
    <div className="card">
      <h2>🎯 Manual Edge Calculator</h2>
      <p className="muted">
        Pick a matchup and type the book's current odds to get the model's edge and
        Kelly stake. Neutral, current-form matchup — works on a static snapshot.
      </p>

      <div className="form-row">
        <label>
          Sport
          <select value={sport} onChange={(e) => setSport(e.target.value)}>
            {SPORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </label>
        <label>
          Market
          <select value={market} onChange={(e) => setMarket(e.target.value)}>
            {markets.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
      </div>

      {market === "prop" ? (
        <div className="form-row">
          <label>
            Player
            <select value={player} onChange={(e) => setPlayer(e.target.value)}>
              {players.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <label>
            Stat
            <select value={stat} onChange={(e) => setStat(e.target.value)}>
              {propStats.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <label>Line<input type="number" step={0.5} value={line} onChange={(e) => setLine(Number(e.target.value))} /></label>
          <label>Over<input type="number" value={overOdds} onChange={(e) => setOverOdds(Number(e.target.value))} /></label>
          <label>Under<input type="number" value={underOdds} onChange={(e) => setUnderOdds(Number(e.target.value))} /></label>
        </div>
      ) : (
        <>
          <div className="form-row">
            <label>
              {isTennis ? "Player A" : "Home"}
              <select value={sideA} onChange={(e) => setSideA(e.target.value)}>
                {entities.map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            <label>
              {isTennis ? "Player B" : "Away"}
              <select value={sideB} onChange={(e) => setSideB(e.target.value)}>
                {entities.map((x) => <option key={x} value={x}>{x}</option>)}
              </select>
            </label>
            {isTennis && (
              <label>
                Surface
                <select value={surface} onChange={(e) => setSurface(e.target.value)}>
                  <option value="">(proxy)</option>
                  <option value="hard">hard</option>
                  <option value="clay">clay</option>
                  <option value="grass">grass</option>
                  <option value="carpet">carpet</option>
                </select>
              </label>
            )}
          </div>
          <div className="form-row">
            {market === "totals" ? (
              <>
                <label>Total<input type="number" step={0.5} value={line} onChange={(e) => setLine(Number(e.target.value))} /></label>
                <label>Over<input type="number" value={overOdds} onChange={(e) => setOverOdds(Number(e.target.value))} /></label>
                <label>Under<input type="number" value={underOdds} onChange={(e) => setUnderOdds(Number(e.target.value))} /></label>
              </>
            ) : market === "spread" ? (
              <>
                <label>Handicap (A)<input type="number" step={0.5} value={line} onChange={(e) => setLine(Number(e.target.value))} /></label>
                <label>A odds<input type="number" value={oddsA} onChange={(e) => setOddsA(Number(e.target.value))} /></label>
                <label>B odds<input type="number" value={oddsB} onChange={(e) => setOddsB(Number(e.target.value))} /></label>
              </>
            ) : (
              <>
                <label>A odds<input type="number" value={oddsA} onChange={(e) => setOddsA(Number(e.target.value))} /></label>
                <label>B odds<input type="number" value={oddsB} onChange={(e) => setOddsB(Number(e.target.value))} /></label>
              </>
            )}
          </div>
        </>
      )}

      <button onClick={calc} disabled={busy || !opts}>{busy ? "…" : "Calculate edge"}</button>
      {error && <div className="error">{error}</div>}

      {best && (
        <div className={best.edge > 0 ? "banner good" : "banner bad"}>
          {best.edge > 0
            ? `✅ Best +EV: ${best.bet} — edge ${pct(best.edge)}, Kelly ${money(best.kelly_stake)}`
            : "No +EV side at these prices."}
        </div>
      )}

      {result && result.length > 0 && (
        <table className="grid">
          <thead>
            <tr><th>Bet</th><th>Odds</th><th>Edge</th><th>Kelly $</th><th>Detail</th></tr>
          </thead>
          <tbody>
            {result.map((e, i) => (
              <tr key={i}>
                <td>{e.bet}</td>
                <td>{odds(e.odds)}</td>
                <td className={e.edge >= 0 ? "pos" : "neg"}>{pct(e.edge)}</td>
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
