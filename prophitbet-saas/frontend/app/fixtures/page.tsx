"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { fixturesApi } from "@/lib/api";
import {
  formatDate,
  formatSASTDateTime,
  formatMatchDay,
  formatMatchKickoff,
  cleanLeagueName,
  getCountryFlag,
  getLeaguePriority,
  sortUpcomingFirst,
  matchesLeagueFilter,
} from "@/lib/utils";

type FixtureItem = {
  id: string;
  league_id: number;
  league_name: string;
  country?: string;
  home_team: string;
  away_team: string;
  match_date: string | null;
  odds_1?: number | null;
  odds_x?: number | null;
  odds_2?: number | null;
  predicted: boolean;
  source_url?: string | null;
  fetched_at?: string | null;
};

export default function FixturesPage() {
  const { token, user } = useAuth();
  const [fixtures, setFixtures] = useState<FixtureItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedLeague, setSelectedLeague] = useState<string>("ALL");
  const [selectedLeagueId, setSelectedLeagueId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");

  const plan = user?.plan || "free";

  useEffect(() => {
    if (!token) return;
    if (plan === "free" && !user?.is_admin) {
      setLoading(false);
      return;
    }
    fixturesApi
      .upcoming(token)
      .then((data) => {
        // Sort upcoming matches soonest first
        const sorted = [...data].sort(sortUpcomingFirst);
        setFixtures(sorted);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [token, plan, user?.is_admin]);

  // Distinct leagues for filter pills
  const availableLeagues = useMemo(() => {
    const map = new Map<string, { name: string; country?: string; id: number; count: number; priority: number }>();
    for (const f of fixtures) {
      const key = `${f.league_name || "Other"}__${f.country || ""}`;
      const existing = map.get(key);
      if (existing) {
        existing.count++;
      } else {
        map.set(key, {
          name: f.league_name?.trim() || "Other",
          country: f.country,
          id: f.league_id,
          count: 1,
          priority: getLeaguePriority(f.league_name || "Other", f.country),
        });
      }
    }
    return [...map.values()].sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return a.name.localeCompare(b.name);
    });
  }, [fixtures]);

  // Filter fixtures
  const filteredFixtures = useMemo(() => {
    return fixtures.filter((f) => {
      if (selectedLeague !== "ALL") {
        if (!matchesLeagueFilter(f, selectedLeague, selectedLeagueId)) {
          return false;
        }
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const homeMatch = f.home_team.toLowerCase().includes(q);
        const awayMatch = f.away_team.toLowerCase().includes(q);
        const leagueMatch = (f.league_name || "").toLowerCase().includes(q);
        if (!homeMatch && !awayMatch && !leagueMatch) return false;
      }
      return true;
    });
  }, [fixtures, selectedLeague, selectedLeagueId, searchQuery]);

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-50">Upcoming Fixtures</h1>
            <p className="text-zinc-400 text-sm mt-1">
              Upcoming provider schedules. Kickoff times are shown in South Africa time (SAST, UTC+2) and may change.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link href="/predictions" className="btn-primary !py-2 !px-4 text-xs inline-flex items-center gap-1.5 shadow-sm">
              <span>🎯</span>
              <span>Predictions Hub</span>
            </Link>
          </div>
        </div>

        {plan === "free" && !user?.is_admin ? (
          <div className="card text-center py-12 bg-zinc-900 border-zinc-700">
            <p className="text-zinc-400 mb-3 text-sm">Fixture tracking requires a Pro or Elite plan.</p>
            <Link href="/billing" className="btn-primary text-xs !py-2 !px-4">
              Upgrade to Pro
            </Link>
          </div>
        ) : loading ? (
          <div className="text-center py-16 text-zinc-400">
            <div className="inline-block animate-spin rounded-full h-7 w-7 border-b-2 border-emerald-500 mb-2" />
            <p className="text-xs">Loading upcoming fixtures...</p>
          </div>
        ) : error ? (
          <div className="card bg-rose-500/10 border-rose-500/30 text-rose-400 text-sm p-4">{error}</div>
        ) : (
          <div className="card !p-5 bg-zinc-900/90 border-zinc-700/90 shadow-md space-y-4">
            {/* Search and Filters */}
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
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
                    placeholder="Search fixture or team (e.g. Arsenal, Real Madrid, Inter)..."
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

                <div className="text-xs text-zinc-400 whitespace-nowrap self-center">
                  Showing <span className="font-semibold text-zinc-200">{filteredFixtures.length}</span> upcoming fixtures
                </div>
              </div>

              {/* League Pills */}
              {availableLeagues.length > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap pt-1 border-t border-zinc-800/70">
                  <span className="text-xs text-zinc-400 font-medium mr-1">League:</span>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedLeague("ALL");
                      setSelectedLeagueId(null);
                    }}
                    className={`px-2.5 py-0.5 text-xs rounded-full border transition ${
                      selectedLeague === "ALL"
                        ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                        : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    All ({fixtures.length})
                  </button>
                  {availableLeagues.map((l) => {
                    const active = selectedLeague === l.name && (!selectedLeagueId || selectedLeagueId === l.id);
                    return (
                      <button
                        key={`${l.name}_${l.country || ""}`}
                        type="button"
                        onClick={() => {
                          if (active) {
                            setSelectedLeague("ALL");
                            setSelectedLeagueId(null);
                          } else {
                            setSelectedLeague(l.name);
                            setSelectedLeagueId(l.id);
                          }
                        }}
                        className={`px-2.5 py-0.5 text-xs rounded-full border transition inline-flex items-center gap-1.5 ${
                          active
                            ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                            : "bg-zinc-800/60 border-zinc-700/60 text-zinc-300 hover:text-zinc-100"
                        }`}
                      >
                        <span>{getCountryFlag(l.country || l.name)}</span>
                        <span>{cleanLeagueName(l.name, l.country)}</span>
                        <span className="text-[10px] opacity-70">({l.count})</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Table */}
            {filteredFixtures.length === 0 ? (
              <div className="text-center py-12 text-zinc-400">
                <p className="text-sm text-zinc-300 font-medium">No matching upcoming fixtures</p>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedLeague("ALL");
                    setSelectedLeagueId(null);
                    setSearchQuery("");
                  }}
                  className="btn-secondary !py-1.5 !px-3 text-xs mt-2"
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
                      <th className="px-4 py-2.5 min-w-[140px]">League</th>
                      <th className="px-4 py-2.5 min-w-[130px] whitespace-nowrap">Date / Kickoff</th>
                      <th className="px-3 py-2.5 text-center min-w-[60px]">1</th>
                      <th className="px-3 py-2.5 text-center min-w-[60px]">X</th>
                      <th className="px-3 py-2.5 text-center min-w-[60px]">2</th>
                      <th className="px-4 py-2.5 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-700/40">
                    {filteredFixtures.map((f) => (
                      <tr key={f.id} className="hover:bg-zinc-800/30 transition group">
                        <td className="px-4 py-3 font-medium text-zinc-100">
                          <span>{f.home_team}</span>
                          <span className="text-zinc-500 text-xs px-1.5 font-normal">vs</span>
                          <span>{f.away_team}</span>
                        </td>
                        <td className="px-4 py-3 text-xs text-zinc-300">
                          <div className="flex items-center gap-1.5">
                            <span>{getCountryFlag(f.country || f.league_name)}</span>
                            <span className="font-medium text-zinc-200">{cleanLeagueName(f.league_name, f.country)}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-xs text-zinc-400">
                          <div className="font-medium text-zinc-200">{formatSASTDateTime(f.match_date)}</div>
                          {f.source_url && <a href={f.source_url} target="_blank" rel="noopener noreferrer" className="text-emerald-400 underline">Schedule source</a>}
                          {f.fetched_at && <div className="text-[11px] text-zinc-500">Checked {formatSASTDateTime(f.fetched_at)}</div>}
                        </td>
                        <td className="px-3 py-3 text-center text-xs font-mono text-zinc-300">
                          {f.odds_1 ? f.odds_1.toFixed(2) : "—"}
                        </td>
                        <td className="px-3 py-3 text-center text-xs font-mono text-zinc-300">
                          {f.odds_x ? f.odds_x.toFixed(2) : "—"}
                        </td>
                        <td className="px-3 py-3 text-center text-xs font-mono text-zinc-300">
                          {f.odds_2 ? f.odds_2.toFixed(2) : "—"}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-right">
                          <div className="flex items-center justify-end gap-2">
                            <Link
                              href={`/live?matchId=${f.id}`}
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25 transition"
                              title="Open match details"
                            >
                              <span>Details</span>
                            </Link>
                            {f.predicted ? (
                              <Link
                                href={`/predictions?league=${encodeURIComponent(f.league_name)}&leagueId=${f.league_id}`}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/25 transition"
                              >
                                <span>Predicted</span>
                                <span>→</span>
                              </Link>
                            ) : (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs text-zinc-400 bg-zinc-800 border border-zinc-700">
                                Scheduled
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
