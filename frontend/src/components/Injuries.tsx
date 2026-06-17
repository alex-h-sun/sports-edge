import type { Injury } from "../api";

export function Injuries({ data }: { data: { nba: Injury[]; nhl: Injury[] } | null }) {
  if (!data) return null;
  const sports: Array<"nba" | "nhl"> = ["nba", "nhl"];

  return (
    <div className="card">
      <h2>🏥 Injury Report</h2>
      <div className="two-col">
        {sports.map((sport) => (
          <div key={sport}>
            <h3>
              {sport.toUpperCase()} ({data[sport].length})
            </h3>
            <div className="scroll">
              <table className="grid small">
                <thead>
                  <tr><th>Player</th><th>Team</th><th>Status</th><th>Injury</th></tr>
                </thead>
                <tbody>
                  {data[sport].map((p, i) => (
                    <tr key={i}>
                      <td>{p.player_name}</td>
                      <td>{p.team_name}</td>
                      <td>{p.status}</td>
                      <td className="muted">{p.injury_type}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
