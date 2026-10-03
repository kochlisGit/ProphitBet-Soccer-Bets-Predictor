"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { marketName, marketPick, MARKET_OPTIONS } from "@/lib/markets";
import MarketProbabilities from "@/components/MarketProbabilities";
import { groupMatchPredictions } from "@/lib/prediction-groups";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { predictionsApi, leaguesApi } from "@/lib/api";
import {
  formatDate,
  formatMatchDay,
  formatMatchKickoff,
  cleanLeagueName,
  getCountryFlag,
  getLeaguePriority,
  matchesLeagueFilter,
  resultColor,
  TOP_LEAGUES,
  sortUpcomingFirst,
} from "@/lib/utils";

type ProbabilityMap = {
  H?: number;
  D?: number;
  A?: number;
  [key: string]: number | undefined;
};

type PredictionItem = {
  id: string;
  league_id: number;
  league_name: string;
  country?: string;
  home_team: string;
  away_team: string;
  match_date: string | null;
  market_type?: string;
  created_at?: string;
  predicted_result: string;
  probabilities?: ProbabilityMap | null;
  actual_result?: string | null;
  is_correct?: boolean | null;
};

function getWinningProb(p: PredictionItem): { label: string; prob: number } {
  if (!p.probabilities) return { label: p.predicted_result, prob: 0 };
  const val = p.probabilities[p.predicted_result];
  return {
    label: p.predicted_result,
    prob: typeof val === "number" ? Math.round(val * 100) : 0,
  };
}

export default function DashboardPage() {
  const { token, user } = useAuth();
  const [predictionRows, setPredictions] = useState<PredictionItem[]>([]);
  const predictions = useMemo(() => groupMatchPredictions(predictionRows), [predictionRows]);
  const [leagues, setLeagues] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(new Set());

  // Quick filters on Dashboard
  const [selectedLeague, setSelectedLeague] = useState<string>("ALL");
  const [selectedPick, setSelectedPick] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  useEffect(() => {
    if (!token) return;
    Promise.all([
      predictionsApi.today(token).catch(() => ({ predictions: [] })),
      leaguesApi.list(token).catch(() => []),
    ]).then(([predData, lgData]) => {
      // Sort upcoming matches soonest first
      const sorted = (predData.predictions || []).sort(sortUpcomingFirst);
      setPredictions(sorted);
      setLeagues(lgData || []);
      setLoading(false);
    });
  }, [token]);

  // Featured Top Competitions
  const featuredLeagues = useMemo(() => {
    const list: any[] = [];
    for (const top of TOP_LEAGUES.slice(0, 6)) {
      const match = leagues.find(
        (l) =>
          l.country.toLowerCase() === top.country.toLowerCase() &&
          (l.name.toLowerCase().replace(/[\s-_]/g, "") ===
            top.name.toLowerCase().replace(/[\s-_]/g, "") ||
            l.name.toLowerCase().includes(top.name.toLowerCase()))
      );
      if (match) {
        list.push({ ...match, rank: top.rank, flag: top.flag, displayName: top.displayName });
      }
    }
    return list.sort((a, b) => a.rank - b.rank);
  }, [leagues]);

  // Extract available leagues in predictions with country disambiguation
  const predLeagues = useMemo(() => {
    const map = new Map<string, { name: string; country?: string; id: number; count: number; priority: number }>();
    for (const p of predictions) {
      const key = String(p.league_id);
      const existing = map.get(key);
      if (existing) {
        existing.count++;
        if (!existing.country && p.country) existing.country = p.country;
      } else {
        map.set(key, {
          name: p.league_name?.trim() || "Other",
          country: p.country,
          id: p.league_id,
          count: 1,
          priority: getLeaguePriority(p.league_name || "", p.country),
        });
      }
    }
    return [...map.values()].sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return cleanLeagueName(a.name, a.country).localeCompare(cleanLeagueName(b.name, b.country));
    });
  }, [predictions]);

  // Filter predictions
  const filteredPredictions = useMemo(() => {
    return predictions.filter((p) => {
      // League filter
      if (!matchesLeagueFilter(p, selectedLeague)) {
        return false;
      }

      // Pick filter
      if (selectedPick !== "ALL" && !p.markets.some(m => (m.market_type || "result") === "result" && m.predicted_result === selectedPick)) {
        return false;
      }

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const homeMatch = p.home_team.toLowerCase().includes(q);
        const awayMatch = p.away_team.toLowerCase().includes(q);
        const leagueMatch = (p.league_name || "").toLowerCase().includes(q);
        if (!homeMatch && !awayMatch && !leagueMatch) return false;
      }

      return true;
    });
  }, [predictions, selectedLeague, selectedPick, searchQuery]);

  // Executive KPI summary
  const kpi = useMemo(() => {
    let home = 0;
    let draw = 0;
    let away = 0;
    let upcoming = 0;
    const now = new Date();
    const nextWeek = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);

    for (const p of predictions) {
      if (p.predicted_result === "H") home++;
      else if (p.predicted_result === "D") draw++;
      else if (p.predicted_result === "A") away++;

      if (p.match_date) {
        const mDate = new Date(p.match_date);
        if (mDate >= now && mDate <= nextWeek) upcoming++;
      }
    }

    const totalCount = predictions.length;
    return {
      total: totalCount,
      home,
      homePct: totalCount > 0 ? Math.round((home / totalCount) * 100) : 0,
      draw,
      drawPct: totalCount > 0 ? Math.round((draw / totalCount) * 100) : 0,
      away,
      awayPct: totalCount > 0 ? Math.round((away / totalCount) * 100) : 0,
      upcoming,
    };
  }, [predictions]);

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Welcome Header & Fast Action Links */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="text-2xl font-bold tracking-tight text-zinc-50">
                Welcome back{user?.name ? `, ${user.name}` : ""}
              </h1>
              <span className="badge bg-emerald-500/15 text-emerald-300 border-emerald-500/30 text-xs uppercase font-bold">
                {user?.plan || "Elite"}
              </span>
            </div>
            <p className="text-zinc-400 text-sm mt-1">
              Active match intelligence, popular league tracking, and AI-predicted outcomes.
            </p>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            <Link
              href="/predictions"
              className="btn-primary !py-2 !px-4 text-sm inline-flex items-center gap-2 shadow-sm"
            >
              <span>🎯</span>
              <span>Predictions Hub ({predictions.length})</span>
            </Link>
            <Link
              href="/leagues"
              className="btn-secondary !py-2 !px-3.5 text-sm inline-flex items-center gap-1.5"
            >
              <span>🏆</span>
              <span>All Leagues ({leagues.length})</span>
            </Link>
          </div>
        </div>

        {/* 4 Executive KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 lg:gap-4">
          <div className="card !p-4 bg-zinc-800/50 border-zinc-700/80">
            <div className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
              Total Predictions (7d)
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-2xl font-bold text-zinc-50">{kpi.total}</span>
              <span className="text-xs text-zinc-400">{predLeagues.length} leagues</span>
            </div>
            <div className="mt-1.5 text-xs text-emerald-400/90 font-medium">
              {kpi.upcoming > 0 ? `⚡ ${kpi.upcoming} matches this week` : "Live dataset"}
            </div>
          </div>

          <div className="card !p-4 bg-zinc-800/50 border-zinc-700/80">
            <div className="text-xs font-medium text-zinc-400 uppercase tracking-wider flex items-center justify-between">
              <span>Home Win Picks</span>
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-2xl font-bold text-emerald-400">{kpi.home}</span>
              <span className="text-sm font-semibold text-zinc-400">{kpi.homePct}%</span>
            </div>
            <div className="w-full bg-zinc-700/60 h-1.5 rounded-full mt-2 overflow-hidden">
              <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${kpi.homePct}%` }} />
            </div>
          </div>

          <div className="card !p-4 bg-zinc-800/50 border-zinc-700/80">
            <div className="text-xs font-medium text-zinc-400 uppercase tracking-wider flex items-center justify-between">
              <span>Draw Picks</span>
              <span className="h-2 w-2 rounded-full bg-amber-400" />
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-2xl font-bold text-amber-400">{kpi.draw}</span>
              <span className="text-sm font-semibold text-zinc-400">{kpi.drawPct}%</span>
            </div>
            <div className="w-full bg-zinc-700/60 h-1.5 rounded-full mt-2 overflow-hidden">
              <div className="bg-amber-500 h-full rounded-full" style={{ width: `${kpi.drawPct}%` }} />
            </div>
          </div>

          <div className="card !p-4 bg-zinc-800/50 border-zinc-700/80">
            <div className="text-xs font-medium text-zinc-400 uppercase tracking-wider flex items-center justify-between">
              <span>Away Win Picks</span>
              <span className="h-2 w-2 rounded-full bg-rose-400" />
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-2xl font-bold text-rose-400">{kpi.away}</span>
              <span className="text-sm font-semibold text-zinc-400">{kpi.awayPct}%</span>
            </div>
            <div className="w-full bg-zinc-700/60 h-1.5 rounded-full mt-2 overflow-hidden">
              <div className="bg-rose-500 h-full rounded-full" style={{ width: `${kpi.awayPct}%` }} />
            </div>
          </div>
        </div>

        {/* ⭐ TOP COMPETITIONS PINNED ROW */}
        {featuredLeagues.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-amber-400 text-sm">⭐</span>
                <h2 className="text-base font-bold text-zinc-100 tracking-tight">
                  Top Competitions Quick Access
                </h2>
              </div>
              <Link href="/leagues" className="text-xs text-emerald-400 hover:underline font-medium">
                View all {leagues.length} leagues →
              </Link>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
              {featuredLeagues.map((fl) => (
                <div
                  key={fl.id}
                  className="card !p-3 bg-zinc-800/40 hover:bg-zinc-800/70 border-zinc-700/80 hover:border-emerald-500/40 transition flex flex-col justify-between group shadow-sm"
                >
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-base">{fl.flag}</span>
                      <h3 className="text-xs font-bold text-zinc-100 truncate group-hover:text-emerald-400 transition">
                        {fl.displayName || cleanLeagueName(fl.name)}
                      </h3>
                    </div>
                    <p className="text-[11px] text-zinc-400 mt-0.5">{fl.country}</p>
                  </div>

                  <div className="mt-2.5 pt-2 border-t border-zinc-800 flex items-center gap-1.5">
                    <Link
                      href={`/predictions?league=${encodeURIComponent(fl.name)}&leagueId=${fl.id}`}
                      className="flex-1 text-center py-1 rounded text-[11px] font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition"
                      title={`View ${fl.displayName} predictions`}
                    >
                      Predictions
                    </Link>
                    <Link
                      href={`/leagues/${fl.id}`}
                      className="text-center py-1 px-1.5 rounded text-[11px] text-zinc-400 hover:text-zinc-200 bg-zinc-800 hover:bg-zinc-700 transition"
                      title="Match table & data"
                    >
                      Data
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* RECENT & UPCOMING PREDICTIONS TABLE */}
        <div className="card !p-5 bg-zinc-900/90 border-zinc-700/90 shadow-md space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold text-zinc-100 tracking-tight flex items-center gap-2">
                <span>Recent & Upcoming Predictions</span>
                <span className="badge bg-zinc-800 text-zinc-300 border-zinc-700 text-xs font-normal">
                  {filteredPredictions.length} matches
                </span>
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                One entry per match. Expand its markets to see all available predictions.
              </p>
            </div>

            <Link
              href="/predictions"
              className="text-xs text-emerald-400 hover:text-emerald-300 font-semibold self-start sm:self-auto flex items-center gap-1"
            >
              <span>Explore All in Predictions Hub</span>
              <span>&rarr;</span>
            </Link>
          </div>

          {/* Search and Filter Row */}
          <div className="pt-2 border-t border-zinc-800 space-y-3">
            <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
              {/* Live Search */}
              <div className="relative flex-1">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-zinc-400">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Quick search match or team (e.g. Southampton, Watford, Betis)..."
                  className="input !pl-9 !py-1.5 text-xs bg-zinc-800/80 border-zinc-700 placeholder-zinc-500 w-full"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery("")}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-200 text-xs"
                  >
                    ✕
                  </button>
                )}
              </div>

              {/* Pick Outcome Filter */}
              <div className="flex items-center gap-1.5 overflow-x-auto">
                <span className="text-xs text-zinc-400 font-medium mr-1 whitespace-nowrap">Pick:</span>
                {[
                  { id: "ALL", label: "All" },
                  { id: "H", label: "Home (H)", color: "text-emerald-400" },
                  { id: "D", label: "Draw (D)", color: "text-amber-400" },
                  { id: "A", label: "Away (A)", color: "text-rose-400" },
                ].map((pick) => {
                  const active = selectedPick === pick.id;
                  return (
                    <button
                      key={pick.id}
                      type="button"
                      onClick={() => setSelectedPick(pick.id)}
                      className={`px-2.5 py-1 text-xs font-semibold rounded-md border transition whitespace-nowrap ${
                        active
                          ? "bg-zinc-700 border-zinc-500 text-zinc-50"
                          : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <span className={active && pick.color ? pick.color : ""}>{pick.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Popular League Filter Pills */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs text-zinc-400 font-medium mr-1">League:</span>

              <button
                type="button"
                onClick={() => setSelectedLeague("ALL")}
                className={`px-2.5 py-0.5 text-xs rounded-full border transition ${
                  selectedLeague === "ALL"
                    ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                    : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                All ({predictions.length})
              </button>

              {predLeagues.map((l) => {
                const active = selectedLeague === `league:${l.id}`;
                const flag = getCountryFlag(l.country || l.name);
                return (
                  <button
                    key={l.id}
                    type="button"
                    onClick={() => setSelectedLeague(active ? "ALL" : `league:${l.id}`)}
                    className={`px-2.5 py-0.5 text-xs rounded-full border transition inline-flex items-center gap-1 ${
                      active
                        ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                        : "bg-zinc-800/60 border-zinc-700/60 text-zinc-300 hover:text-zinc-100"
                    }`}
                  >
                    <span>{flag}</span>
                    <span>{cleanLeagueName(l.name, l.country)}</span>
                    <span className="text-[10px] opacity-70">({l.count})</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Predictions Table */}
          {loading ? (
            <div className="text-center py-12 text-zinc-400">
              <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500 mb-2" />
              <p className="text-xs">Loading latest predictions...</p>
            </div>
          ) : filteredPredictions.length === 0 ? (
            <div className="text-center py-12 text-zinc-400 space-y-2">
              <p className="text-sm text-zinc-300 font-medium">No matching predictions found</p>
              <button
                type="button"
                onClick={() => {
                  setSelectedLeague("ALL");
                  setSelectedPick("ALL");
                  setSearchQuery("");
                }}
                className="btn-secondary !py-1.5 !px-3 text-xs"
              >
                Reset Filters
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs font-semibold text-zinc-400 border-b border-zinc-700/70 uppercase tracking-wider bg-zinc-900/30">
                    <th className="px-4 py-2.5 min-w-[200px]">Match</th>
                    <th className="px-4 py-2.5 min-w-[130px]">League</th>
                    <th className="px-4 py-2.5 whitespace-nowrap min-w-[110px]">Date</th>
                    <th className="px-4 py-2.5 min-w-[100px]">Prediction</th>
                    <th className="px-4 py-2.5 min-w-[170px]">Model Probability</th>
                    <th className="px-4 py-2.5 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-700/40">
                  {filteredPredictions.slice(0, 20).map((p) => {
                    const win = getWinningProb(p);
                    const probs = p.probabilities;
                    const hProb = probs?.H ? Math.round(probs.H * 100) : null;
                    const dProb = probs?.D ? Math.round(probs.D * 100) : null;
                    const aProb = probs?.A ? Math.round(probs.A * 100) : null;

                    return (
                      <Fragment key={p.id}>
                      <tr className="hover:bg-zinc-800/30 transition group [&>td]:align-top">
                        {/* Match */}
                        <td className="px-4 py-3 font-medium text-zinc-100">
                          <Link
                            href={`/predictions/${p.id}`}
                            className="hover:text-emerald-400 transition inline-block"
                          >
                            <span>{p.home_team}</span>
                            <span className="text-zinc-500 text-xs px-1.5 font-normal">vs</span>
                            <span>{p.away_team}</span>
                          </Link>
                          {p.markets.length > 1 && (
                            <button type="button" className="mt-2 block text-xs font-normal text-emerald-400 hover:text-emerald-300"
                              aria-expanded={expandedMatches.has(p.id)} aria-controls={`markets-${p.id}`}
                              onClick={() => setExpandedMatches(previous => {
                                const next = new Set(previous);
                                if (next.has(p.id)) next.delete(p.id); else next.add(p.id);
                                return next;
                              })}>
                              {expandedMatches.has(p.id) ? "Hide" : "View"} {p.markets.length} market predictions {expandedMatches.has(p.id) ? "▴" : "▾"}
                            </button>
                          )}
                        </td>

                        {/* League */}
                        <td className="px-4 py-3">
                          <Link
                            href={`/predictions?league=${encodeURIComponent(p.league_name)}&leagueId=${p.league_id}`}
                            className="inline-flex items-center gap-1.5 text-xs text-zinc-300 hover:text-emerald-400 transition"
                          >
                            <span>{getCountryFlag(p.country || p.league_name)}</span>
                            <span>{cleanLeagueName(p.league_name, p.country)}</span>
                          </Link>
                        </td>

                        {/* Date */}
                        <td className="px-4 py-3 whitespace-nowrap text-xs text-zinc-400">
                          <div className="font-medium text-zinc-300">{formatMatchKickoff(p.match_date)}</div>
                          <div className="text-[11px] text-zinc-500">{formatDate(p.match_date)}</div>
                        </td>

                        {/* Predicted Result Badge */}<td className="px-4 py-3"><span className="text-xs font-semibold text-emerald-300">{marketPick(p.market_type, p.predicted_result)}</span><p className="text-[10px] text-zinc-500">{marketName(p.market_type)}</p></td>

                        {/* Model Probability Bar */}<td className="px-4 py-3"><MarketProbabilities probabilities={p.probabilities}/></td>

                        {/* Action Link */}
                        <td className="px-4 py-3 text-right whitespace-nowrap">
                          <Link
                            href={`/predictions/${p.id}`}
                            className="text-xs text-zinc-400 group-hover:text-emerald-400 group-hover:underline transition font-medium"
                          >
                            Details →
                          </Link>
                        </td>
                      </tr>
                      {p.markets.length > 1 && (
                        <tr id={`markets-${p.id}`} hidden={!expandedMatches.has(p.id)}>
                          <td colSpan={6} className="bg-zinc-950/40 px-4 py-4">
                            <p className="mb-3 text-xs text-zinc-400">Markets for {p.home_team} vs {p.away_team}</p>
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                              {p.markets.map(m => (
                                <div key={m.id} className="rounded-lg border border-zinc-700 bg-zinc-900 p-3">
                                  <div className="flex items-start justify-between gap-3 mb-3">
                                    <div>
                                      <p className="text-xs font-semibold text-emerald-300">{marketPick(m.market_type, m.predicted_result)}</p>
                                      <p className="text-[10px] text-zinc-400 mt-1">{marketName(m.market_type)}</p>
                                    </div>
                                    <Link href={`/predictions/${m.id}`} aria-label={`Details for ${marketName(m.market_type)}: ${p.home_team} vs ${p.away_team}`} className="shrink-0 text-xs text-emerald-400 hover:underline">Details →</Link>
                                  </div>
                                  <MarketProbabilities probabilities={m.probabilities}/>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {filteredPredictions.length > 20 && (
            <div className="pt-3 border-t border-zinc-800 text-center">
              <Link
                href="/predictions"
                className="btn-secondary !py-2 !px-4 text-xs inline-flex items-center gap-1.5"
              >
                <span>View All {filteredPredictions.length} Matches in Predictions Hub</span>
                <span>&rarr;</span>
              </Link>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
