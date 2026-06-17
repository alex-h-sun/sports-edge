import { SPORT_LABELS, MARKET_OPTIONS } from "../constants";

interface Props {
  sports: string[];
  setSports: (s: string[]) => void;
  markets: string[];
  setMarkets: (m: string[]) => void;
  minEdge: number;
  setMinEdge: (n: number) => void;
  allSports: string[];
}

function toggle(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((x) => x !== value) : [...list, value];
}

export function Controls(p: Props) {
  return (
    <div className="card controls">
      <div className="control-group">
        <label>Sports</label>
        <div className="chips">
          {p.allSports.map((s) => (
            <button
              key={s}
              className={p.sports.includes(s) ? "chip on" : "chip"}
              onClick={() => p.setSports(toggle(p.sports, s))}
            >
              {SPORT_LABELS[s] ?? s}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label>Markets</label>
        <div className="chips">
          {MARKET_OPTIONS.map((m) => (
            <button
              key={m}
              className={p.markets.includes(m) ? "chip on" : "chip"}
              onClick={() => p.setMarkets(toggle(p.markets, m))}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label>Min edge: {(p.minEdge * 100).toFixed(0)}%</label>
        <input
          type="range"
          min={0}
          max={20}
          step={1}
          value={Math.round(p.minEdge * 100)}
          onChange={(e) => p.setMinEdge(Number(e.target.value) / 100)}
        />
      </div>
    </div>
  );
}
