export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    // Always use the same-origin Next.js proxy in the browser. Public env vars
    // are baked into production bundles and can otherwise point at a stale
    // host/port after deployment.
    return "/api/proxy";
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
}

interface FetchOptions extends RequestInit {
  token?: string | null;
}

export async function apiFetch<T = any>(
  path: string,
  options: FetchOptions = {}
): Promise<T> {
  const { token, headers: customHeaders, ...rest } = options;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(customHeaders as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const baseUrl = getApiBaseUrl();
  let res = await fetch(`${baseUrl}${path}`, { credentials: "include", headers, ...rest });

  if (res.status === 401 && token) {
    const newToken = await tryRefreshToken();
    if (newToken) {
      headers["Authorization"] = `Bearer ${newToken}`;
      res = await fetch(`${baseUrl}${path}`, { credentials: "include", headers, ...rest });
    }
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || "Request failed");
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

let onAccessTokenRefreshed: ((accessToken: string) => void) | null = null;

/** Register a callback so AuthContext can stay in sync when apiFetch refreshes the session. */
export function setAccessTokenRefreshHandler(handler: ((accessToken: string) => void) | null) {
  onAccessTokenRefreshed = handler;
}

async function tryRefreshToken(): Promise<string | null> {
  try {
    const baseUrl = getApiBaseUrl();
    const res = await fetch(`${baseUrl}/auth/refresh`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) return null;
    const data = await res.json();
    onAccessTokenRefreshed?.(data.access_token);
    return data.access_token;
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

// Auth
export const authApi = {
  register: (email: string, password: string, name: string) =>
    apiFetch("/auth/register", { method: "POST", body: JSON.stringify({ email, password, name }) }),
  login: (email: string, password: string) =>
    apiFetch("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  refresh: () => apiFetch("/auth/refresh", { method: "POST" }),
  logout: () => apiFetch("/auth/logout", { method: "POST" }),
  me: (token: string) => apiFetch("/auth/me", { token }),
  updateProfile: (
    data: {
      name?: string;
      notify_daily_digest?: boolean;
      notify_training?: boolean;
      notify_weekly_report?: boolean;
    },
    token: string
  ) => apiFetch("/auth/me", { method: "PATCH", body: JSON.stringify(data), token }),
};

// Predictions
export const predictionsApi = {
  today: (token?: string) => apiFetch("/predictions/today", { token }),
  byLeague: (leagueId: number, page: number, token: string) =>
    apiFetch(`/predictions/league/${leagueId}?page=${page}`, { token }),
  history: (
    days: number,
    page: number,
    token: string,
    perPage: number = 50,
    leagueId?: number | null,
    league?: string | null,
    marketType: string = "all",
    groupMatches: boolean = false
  ) => {
    let url = `/predictions/history?days=${days}&page=${page}&per_page=${perPage}&market_type=${encodeURIComponent(marketType)}`;
    if (groupMatches) url += "&group_matches=true";
    if (leagueId) url += `&league_id=${leagueId}`;
    if (league && league !== "ALL") url += `&league=${encodeURIComponent(league)}`;
    return apiFetch(url, { token });
  },
  get: (id: string, token: string) => apiFetch(`/predictions/${id}`, { token }),
};

// Leagues
export const leaguesApi = {
  list: (token?: string) => apiFetch("/leagues", { token }),
  get: (id: number, token?: string) => apiFetch(`/leagues/${id}`, { token }),
  stats: (id: number, token?: string) => apiFetch(`/leagues/${id}/stats`, { token }),
  table: (id: number, page: number, token?: string) =>
    apiFetch(`/leagues/${id}/table?page=${page}`, { token }),
};

// Models
export const modelsApi = {
  list: (token: string) => apiFetch("/models", { token }),
  train: (data: any, token: string) =>
    apiFetch("/models/train", { method: "POST", body: JSON.stringify(data), token }),
  autoTune: (data: any, token: string) =>
    apiFetch("/models/auto-tune", { method: "POST", body: JSON.stringify(data), token }),
  status: (jobId: string, token: string) => apiFetch(`/models/${jobId}/status`, { token }),
  get: (id: string, token: string) => apiFetch(`/models/${id}`, { token }),
  delete: (id: string, token: string) =>
    apiFetch(`/models/${id}`, { method: "DELETE", token }),
};

// Analysis & Quantitative Backtesting
export const analysisApi = {
  types: (token: string) => apiFetch("/analysis/types", { token }),
  run: (data: any, token: string) =>
    apiFetch("/analysis", { method: "POST", body: JSON.stringify(data), token }),
  backtestPresets: (token?: string) => apiFetch("/analysis/backtest/presets", { token }),
  backtest: (filters: any, token?: string) =>
    apiFetch("/analysis/backtest", { method: "POST", body: JSON.stringify(filters), token }),
};

// Fixtures
export const fixturesApi = {
  upcoming: (token: string, leagueId?: number, league?: string, limit?: number) => {
    const q = new URLSearchParams();
    if (leagueId) q.set("league_id", String(leagueId));
    if (league && league !== "ALL") q.set("league", league);
    if (limit) q.set("limit", String(limit));
    const qs = q.toString();
    return apiFetch(`/fixtures/upcoming${qs ? `?${qs}` : ""}`, { token });
  },
  byLeague: (leagueId: number, token: string) =>
    apiFetch(`/fixtures/league/${leagueId}`, { token }),
};

// Billing
export const billingApi = {
  checkout: (plan: string, token: string) =>
    apiFetch("/billing/checkout", { method: "POST", body: JSON.stringify({ plan }), token }),
  portal: (token: string) => apiFetch("/billing/portal", { token }),
  usage: (token: string) => apiFetch("/billing/usage", { token }),
};

// MiroFish Swarm Intelligence
export const mirofishApi = {
  status: (token?: string) => apiFetch("/mirofish/status", { token }),
  stats: (token?: string) => apiFetch("/mirofish/stats", { token }),
  simulations: (params?: { limit?: number; consensus?: string; search?: string }, token?: string) => {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.consensus) q.set("consensus", params.consensus);
    if (params?.search) q.set("search", params.search);
    const qs = q.toString();
    return apiFetch(`/mirofish/simulations${qs ? `?${qs}` : ""}`, { token });
  },
  sandboxSimulate: (data: any, token?: string) =>
    apiFetch("/mirofish/sandbox/simulate", {
      method: "POST",
      body: JSON.stringify(data),
      token,
    }),
  runContinuousTest: (data?: { batch_size?: number; stress_mode?: string }, token?: string) =>
    apiFetch("/mirofish/continuous-test/run", {
      method: "POST",
      body: JSON.stringify(data || {}),
      token,
    }),
  getSimulation: (predictionId: string, token?: string) =>
    apiFetch(`/mirofish/predictions/${predictionId}/simulation`, { token }),
  simulate: (predictionId: string, forceRecompute: boolean = false, token?: string) =>
    apiFetch(`/mirofish/predictions/${predictionId}/simulate`, {
      method: "POST",
      body: JSON.stringify({ force_recompute: forceRecompute }),
      token,
    }),
  calibration: (token?: string) => apiFetch("/mirofish/calibration", { token }),
  tacticalIntel: (teamName: string, token?: string) =>
    apiFetch(`/mirofish/tactical-intel/${encodeURIComponent(teamName)}`, { token }),
  seedTacticalIntel: (token: string) =>
    apiFetch("/mirofish/tactical-intel/seed", { method: "POST", token }),
};

// Closing Line Value (CLV) & Odds Drift
export const clvApi = {
  audit: (limit: number = 100, token: string) =>
    apiFetch(`/analysis/clv/audit?limit=${limit}`, { token }),
  drift: (predictionId: string, token: string) =>
    apiFetch(`/analysis/clv/drift/${predictionId}`, { token }),
};

// Bet Slip & Bankroll Journal
export const betslipApi = {
  getSlips: (params?: { status?: string; limit?: number }, token?: string) => {
    const q = new URLSearchParams();
    if (params?.status) q.set("status", params.status);
    if (params?.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return apiFetch(`/betslip${qs ? `?${qs}` : ""}`, { token });
  },
  createSlip: (data: any, token: string) =>
    apiFetch("/betslip", { method: "POST", body: JSON.stringify(data), token }),
  updateSlip: (slipId: string, data: any, token: string) =>
    apiFetch(`/betslip/${slipId}`, { method: "PATCH", body: JSON.stringify(data), token }),
  deleteSlip: (slipId: string, token: string) =>
    apiFetch(`/betslip/${slipId}`, { method: "DELETE", token }),
  autoSettle: (token: string) =>
    apiFetch("/betslip/auto-settle", { method: "POST", token }),
};

// Value Alerts & Bot Dispatcher
export const alertApi = {
  preview: (minEv: number = 4.0, token: string) =>
    apiFetch(`/admin/alerts/preview?min_ev=${minEv}`, { token }),
  dispatch: (data: { webhook_url: string; min_ev_pct?: number; platform?: string }, token: string) =>
    apiFetch("/admin/alerts/dispatch", { method: "POST", body: JSON.stringify(data), token }),
};

// South African Betting Markets & Perfect Bet Slip Engine
export const saMarketsApi = {
  buildAISlip: (data: {bookmaker: string; legs: number; min_probability: number; strategy: string; draft: boolean; market_type?: string}, token: string) =>
    apiFetch('/sa-markets/ai-slip', { method: 'POST', body: JSON.stringify(data), token }),
  getOdds: (league?: string, limit: number = 50, token?: string | null) => {
    const q = new URLSearchParams();
    if (league) q.set("league", league);
    q.set("limit", String(limit));
    return apiFetch(`/sa-markets/odds?${q.toString()}`, { token: token || undefined });
  },
  getPerfectSlips: (bankroll: number = 1000, token?: string | null) =>
    apiFetch(`/sa-markets/perfect-slips?bankroll=${bankroll}`, { token: token || undefined }),
  saveSlip: (data: {
    match_title: string;
    league_name?: string;
    selection: string;
    odds_taken: number;
    stake_amount: number;
    bookmaker?: string;
    notes?: string;
  }, token: string) =>
    apiFetch("/sa-markets/slips/save", { method: "POST", body: JSON.stringify(data), token }),
  getArbitrage: (bankroll: number = 1000, league?: string, token?: string | null) => {
    const q = new URLSearchParams({ bankroll: String(bankroll) });
    if (league && league !== "ALL") q.set("league", league);
    return apiFetch(`/sa-markets/arbitrage?${q.toString()}`, { token: token || undefined });
  },
  getValueBets: (minEv: number = 3.0, bankroll: number = 2000, league?: string, token?: string | null) => {
    const q = new URLSearchParams({ min_ev: String(minEv), bankroll: String(bankroll) });
    if (league && league !== "ALL") q.set("league", league);
    return apiFetch(`/sa-markets/value-bets?${q.toString()}`, { token: token || undefined });
  },
  getMargins: (token?: string | null) =>
    apiFetch("/sa-markets/margins", { token: token || undefined }),
  syncOdds: (token?: string | null) =>
    apiFetch("/sa-markets/sync", { method: "POST", token: token || undefined }),
};

// Live Match Watch & Commentary API
export const liveMatchApi = {
  getMatches: (league?: string, status?: string, limit: number = 60, token?: string | null) => {
    const q = new URLSearchParams();
    if (league && league !== "ALL") q.set("league", league);
    if (status && status !== "ALL") q.set("status", status);
    q.set("limit", String(limit));
    return apiFetch(`/live/matches?${q.toString()}`, { token: token || undefined });
  },
  getTimeline: (matchId: string, token?: string | null) =>
    apiFetch(`/live/matches/${matchId}/timeline`, { token: token || undefined }),
};

// Match Results, Prediction Audit & System Learning API
export const resultsApi = {
  list: (
    params?: {
      league?: string;
      outcome?: string;
      accuracy?: string;
      days?: number;
      page?: number;
      per_page?: number;
    },
    token?: string | null
  ) => {
    const q = new URLSearchParams();
    if (params?.league && params.league !== "ALL") q.set("league", params.league);
    if (params?.outcome && params.outcome !== "ALL") q.set("outcome", params.outcome);
    if (params?.accuracy && params.accuracy !== "ALL") q.set("accuracy", params.accuracy);
    if (params?.days) q.set("days", String(params.days));
    if (params?.page) q.set("page", String(params.page));
    if (params?.per_page) q.set("per_page", String(params.per_page));
    const qs = q.toString();
    return apiFetch(`/results${qs ? `?${qs}` : ""}`, { token: token || undefined });
  },
  summary: (days: number = 90, token?: string | null) => {
    return apiFetch(`/results/summary?days=${days}`, { token: token || undefined });
  },
  triggerLearning: (token?: string | null) => {
    return apiFetch("/results/learn", {
      method: "POST",
      token: token || undefined,
    });
  },
};




