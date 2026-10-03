type MatchPrediction = {
  id: string;
  league_id: number;
  home_team: string;
  away_team: string;
  match_date: string | null;
  market_type?: string;
  created_at?: string;
};

// Group markets, not separate fixtures. Undated rows have no reliable match identity.
export function groupMatchPredictions<T extends MatchPrediction>(rows: T[]): (T & { markets: T[] })[] {
  const matches = new Map<string, Map<string, T>>();
  for (const row of rows) {
    const date = row.match_date ? Date.parse(row.match_date) : NaN;
    const key = Number.isFinite(date)
      ? JSON.stringify([row.league_id, row.home_team.trim().toLowerCase(), row.away_team.trim().toLowerCase(), date])
      : row.id;
    const markets = matches.get(key) || new Map<string, T>();
    const market = row.market_type || "result";
    const previous = markets.get(market);
    if (!previous || (row.created_at || "") > (previous.created_at || "")) markets.set(market, row);
    matches.set(key, markets);
  }
  return [...matches.values()].map(byMarket => {
    const markets = [...byMarket.values()];
    const primary = byMarket.get("result") || markets[0];
    return { ...primary, markets };
  });
}
