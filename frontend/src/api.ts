// Typed client for the Starlette JSON API. All requests send the session cookie.

export interface Meta {
  db_label: string;
  bankroll: number;
  min_edge: number;
  sports: string[];
  default_sports: string[];
  snapshot_ts: string | null;
  auth_required: boolean;
}

export interface Edge {
  sport: string;
  market: string;
  game: string;
  bet: string;
  odds: number;
  edge: number;
  kelly_stake: number;
  model_prob?: number;
  fair_prob?: number;
  model_total?: number;
  book_total?: number;
  model_pred?: number;
  book_line?: number;
}

export interface EdgesResponse {
  edges: Edge[];
  count: number;
  errors: string[];
  sports: string[];
  min_edge: number;
}

export interface Injury {
  team_name: string;
  player_name: string;
  position: string;
  status: string;
  injury_type: string;
  short_comment: string;
}

export interface Quota {
  remaining: number;
  used: number;
  fetched_at: string;
}

export interface BankrollSummary {
  current: number;
  total_return_pct: number;
  hit_rate: number;
  n_settled: number;
  max_drawdown_pct: number;
  n_open: number;
  n_void: number;
}

export type LedgerRow = Record<string, string | number | null>;

export interface Bankroll {
  start: number;
  summary: BankrollSummary | null;
  curve: { game_date: string; balance: number }[];
  ledger: LedgerRow[];
}

export interface ManualOptions {
  sport: string;
  entities: string[];
  players: string[];
  prop_stats: string[];
}

export interface EdgeQuery {
  sports: string[];
  minEdge: number;
  markets: string[];
}

export interface PullStage {
  sport: string | null;
  stage: string; // games | injuries | odds | paper
  status: string; // ok | skipped | error
  detail?: string;
}

export interface PullResult {
  duration_s: number;
  sports: string[];
  fetched_odds: boolean;
  stages: PullStage[];
  n_edges: number;
  edge_errors: string[];
  quota: Quota | null;
  paper: BankrollSummary | null;
  published?: { version?: string; error?: string };
}

export interface PullRequest {
  sports?: string[];
  odds?: boolean;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

function edgesQuery(q: EdgeQuery): string {
  const params = new URLSearchParams({
    sports: q.sports.join(","),
    min_edge: String(q.minEdge),
    markets: q.markets.join(","),
  });
  return params.toString();
}

export const api = {
  meta: () => req<Meta>("/api/meta"),
  login: (password: string) =>
    req<{ ok: boolean }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  logout: () => req<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  edges: (q: EdgeQuery) => req<EdgesResponse>(`/api/edges?${edgesQuery(q)}`),
  edgesCsvUrl: (q: EdgeQuery) => `/api/edges/export.csv?${edgesQuery(q)}`,
  injuries: () => req<{ nba: Injury[]; nhl: Injury[] }>("/api/injuries"),
  quota: () => req<{ quota: Quota | null }>("/api/quota"),
  bankroll: () => req<Bankroll>("/api/bankroll"),
  manualOptions: (sport: string) =>
    req<ManualOptions>(`/api/manual/options?sport=${encodeURIComponent(sport)}`),
  manualEdge: (body: Record<string, unknown>) =>
    req<{ edges: Edge[]; count: number }>("/api/manual/edge", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  reload: () => req<{ status: string; version?: string }>("/api/admin/reload", { method: "POST" }),
  pull: (body: PullRequest = {}) =>
    req<PullResult>("/api/pull", { method: "POST", body: JSON.stringify(body) }),
};
