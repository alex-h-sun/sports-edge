// Tiny hand-rolled SVG line chart for the equity curve (no charting dependency).

export function EquityChart({ points }: { points: { game_date: string; balance: number }[] }) {
  if (points.length < 2) return null;

  const W = 600;
  const H = 160;
  const pad = 10;
  const ys = points.map((p) => p.balance);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanY = maxY - minY || 1;

  const x = (i: number) => pad + (i / (points.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - ((v - minY) / spanY) * (H - 2 * pad);

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.balance).toFixed(1)}`)
    .join(" ");
  const up = ys[ys.length - 1] >= ys[0];
  const color = up ? "#3fb950" : "#f85149";

  return (
    <svg className="equity" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Equity curve">
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  );
}
