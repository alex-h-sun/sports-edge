// Display formatting helpers shared across components.

export const pct = (x: number, digits = 1): string => `${(x * 100).toFixed(digits)}%`;

export const money = (x: number): string =>
  `$${x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const odds = (x: number): string => (x > 0 ? `+${x}` : String(x));

interface EdgeLike {
  model_prob?: number;
  fair_prob?: number;
  model_total?: number;
  book_total?: number;
  model_pred?: number;
  book_line?: number;
}

export function edgeDetail(e: EdgeLike): string {
  if (e.model_prob != null && e.fair_prob != null)
    return `Model ${pct(e.model_prob)} vs fair ${pct(e.fair_prob)}`;
  if (e.model_total != null && e.book_total != null)
    return `Model ${e.model_total} vs line ${e.book_total}`;
  if (e.model_pred != null && e.book_line != null)
    return `Model ${e.model_pred} vs line ${e.book_line}`;
  return "";
}
