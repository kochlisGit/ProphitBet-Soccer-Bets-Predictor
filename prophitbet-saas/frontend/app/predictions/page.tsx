"use client";

import { Fragment, Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { marketName, marketPick, MARKET_OPTIONS } from "@/lib/markets";
import MarketProbabilities from "@/components/MarketProbabilities";
import { groupMatchPredictions } from "@/lib/prediction-groups";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { predictionsApi } from "@/lib/api";
import {
  formatDate,
  formatMatchDay,
  formatMatchKickoff,
  cleanLeagueName,
  getLeaguePriority,
  getCountryFlag,
  matchesLeagueFilter,
  resultColor,
  sortUpcomingFirst,
} from "@/lib/utils";

type ProbabilityMap = {
  H?: number;
  D?: number;
  A?: number;
  [key: string]: number | undefined;
};

type PredictionRow = {
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

type MatchRow = PredictionRow & { markets: PredictionRow[] };
type LeagueGroup = {
  league: string;
  country?: string;
  leagueId: number;
  priority: number;
  items: MatchRow[];
  homeCount: number;
  drawCount: number;
  awayCount: number;
};

function getWinningProb(p: PredictionRow): { label: string; prob: number } {
  if (!p.probabilities) return { label: p.predicted_result, prob: 0 };
  const val = p.probabilities[p.predicted_result];
  return {
    label: p.predicted_result,
    prob: typeof val === "number" ? Math.round(val * 100) : 0,
  };
}

function PredictionsContent() {
  const searchParams = useSearchParams();
  const urlLeague = searchParams.get("league") || "";
  const urlLeagueId = searchParams.get("leagueId") ? Number(searchParams.get("leagueId")) : null;

  const { token } = useAuth();
  const [predictionRows, setPredictions] = useState<PredictionRow[]>([]);
  const predictions = useMemo(() => groupMatchPredictions(predictionRows), [predictionRows]);
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(new Set());
  const [total, setTotal] = useState(0);
  const [perPage, setPerPage] = useState(50);
  const [page, setPage] = useState(1);
  const [days, setDays] = useState(30);
  const [marketType, setMarketType] = useState("all");
  const [loading, setLoading] = useState(true);

  // Client-side interactive filters for quick assessment
  const [selectedLeague, setSelectedLeague] = useState<string>(urlLeague);
  const [selectedPick, setSelectedPick] = useState<string>("ALL"); // ALL, H, D, A
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [collapsedLeagues, setCollapsedLeagues] = useState<Record<string, boolean>>({});

  // Sync urlLeague if param changes
  useEffect(() => {
    if (urlLeague) setSelectedLeague(urlLeague);
  }, [urlLeague]);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    predictionsApi
      .history(days, page, token, perPage, urlLeagueId, undefined, marketType, true)
      .then((data: any) => {
        const sorted = (data.predictions ?? []).sort(sortUpcomingFirst);
        setPredictions(sorted);
        setTotal(data.total ?? 0);
      })
      .catch(() => setPredictions([]))
      .finally(() => setLoading(false));
  }, [token, page, days, perPage, urlLeagueId, marketType]);

  // Extract distinct leagues for filter bar with disambiguated country
  const availableLeagues = useMemo(() => {
    const map = new Map<string, { name: string; country?: string; id: number; priority: number; count: number }>();
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
          priority: getLeaguePriority(p.league_name || "", p.country),
          count: 1,
        });
      }
    }
    return [...map.values()].sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return cleanLeagueName(a.name, a.country).localeCompare(cleanLeagueName(b.name, b.country));
    });
  }, [predictions]);

  // Filter predictions by live search, league filter, and outcome pick
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

      // Search query (teams or league)
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

  // Group filtered predictions by league, keeping popular leagues first
  const leagueGroups = useMemo(() => {
    const map = new Map<string, { id: number; country?: string; items: MatchRow[] }>();
    for (const p of filteredPredictions) {
      const key = p.league_name?.trim() || "Other";
      if (!map.has(key)) map.set(key, { id: p.league_id, country: p.country, items: [] });
      map.get(key)!.items.push(p);
    }

    const groups: LeagueGroup[] = [...map.entries()].map(([league, data]) => {
      const sortedItems = [...data.items].sort(sortUpcomingFirst);
      let homeCount = 0;
      let drawCount = 0;
      let awayCount = 0;
      for (const item of sortedItems) {
        if (item.predicted_result === "H") homeCount++;
        else if (item.predicted_result === "D") drawCount++;
        else if (item.predicted_result === "A") awayCount++;
      }
      return {
        league,
        country: data.country,
        leagueId: data.id,
        priority: getLeaguePriority(league),
        items: sortedItems,
        homeCount,
        drawCount,
        awayCount,
      };
    });

    // Sort popular leagues first
    groups.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return a.league.localeCompare(b.league);
    });

    return groups;
  }, [filteredPredictions]);

  // Overall KPI statistics
  const kpi = useMemo(() => {
    let home = 0;
    let draw = 0;
    let away = 0;
    let upcoming = 0;
    const now = new Date();
    const nextWeek = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000);

    for (const p of filteredPredictions) {
      if (p.predicted_result === "H") home++;
      else if (p.predicted_result === "D") draw++;
      else if (p.predicted_result === "A") away++;

      if (p.match_date) {
        const mDate = new Date(p.match_date);
        if (mDate >= now && mDate <= nextWeek) upcoming++;
      }
    }

    const totalCount = filteredPredictions.length;
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
  }, [filteredPredictions]);

  const toggleAllAccordions = (expand: boolean) => {
    const updated: Record<string, boolean> = {};
    for (const g of leagueGroups) {
      updated[g.league] = !expand;
    }
    setCollapsedLeagues(updated);
  };

  const toggleGroup = (league: string) => {
    setCollapsedLeagues((prev) => ({
      ...prev,
      [league]: !prev[league],
    }));
  };

  const hasNextPage = page * perPage < total;
  const hasPrevPage = page > 1;

  return (
    <div className="space-y-6">
      {/* Header & Main Controls */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-50">Predictions Hub</h1>
            <span className="badge bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
              AI-Powered
            </span>
          </div>
          <p className="text-zinc-400 text-sm mt-1">
            One entry per match. Expand market predictions for probabilities and individual details.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={days}
            onChange={(e) => {
              setDays(Number(e.target.value));
              setPage(1);
            }}
            className="input !w-auto text-sm bg-zinc-800/90 border-zinc-700 text-zinc-200"
            aria-label="Filter by time range"
          >
            <option value={7}>Next 7 days</option>
            <option value={14}>Next 14 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={365}>Full history</option>
          </select>

          <select
            value={perPage}
            onChange={(e) => {
              setPerPage(Number(e.target.value));
              setPage(1);
            }}
            className="input !w-auto text-sm bg-zinc-800/90 border-zinc-700 text-zinc-200"
            aria-label="Records per page"
          >
            <option value={25}>25 per batch</option>
            <option value={50}>50 per batch</option>
            <option value={100}>100 per batch</option>
          </select>

          <Link
            href="/leagues"
            className="btn-secondary !py-2 !px-3.5 text-sm inline-flex items-center gap-1.5"
          >
            <svg className="w-4 h-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
            </svg>
            All Leagues Directory
          </Link>
        </div>
      </div>

      {/* KPI Overview Cards for Quick Assessment */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 lg:gap-4">
        <div className="card !p-4 bg-zinc-800/50 border-zinc-700/80">
          <div className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Total Matches</div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="text-2xl font-bold text-zinc-50">{kpi.total}</span>
            <span className="text-xs text-zinc-400">
              {leagueGroups.length} league{leagueGroups.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="mt-1 text-xs text-emerald-400/90 font-medium">
            {kpi.upcoming > 0 ? `⚡ ${kpi.upcoming} coming up this week` : "Active dataset"}
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

      {/* Interactive Filter and Search Bar */}
      <div className="card !p-4 bg-zinc-900/90 border-zinc-700/90 shadow-md space-y-4">
        {/* Row 1: Search & Pick Outcome Filter */}
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
              placeholder="Search team or match (e.g. Barcelona, Arsenal, Betis)..."
              className="input !pl-9 !py-2 text-sm bg-zinc-800/80 border-zinc-700 placeholder-zinc-500 w-full"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-200 text-xs"
              >
                ✕ Clear
              </button>
            )}
          </div>

          <label className="text-xs text-zinc-400">Prediction market
            <select value={marketType} onChange={e => { setMarketType(e.target.value); setPage(1); setSelectedPick("ALL"); }} className="input mt-1">
              <option value="all">All trained markets</option>
              {MARKET_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          {/* Outcome Filter Chips */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
            <span className="text-xs text-zinc-400 font-medium mr-1 whitespace-nowrap">Pick:</span>
            {[
              { id: "ALL", label: "All Picks" },
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
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition whitespace-nowrap ${
                    active
                      ? "bg-zinc-700 border-zinc-500 text-zinc-50 shadow-sm"
                      : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                  }`}
                >
                  <span className={active && pick.color ? pick.color : ""}>{pick.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Row 2: Popular League Quick Selector Pills */}
        <div className="pt-2 border-t border-zinc-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-zinc-400 font-medium mr-1">Leagues:</span>

            {/* All Leagues Pill */}
            <button
              type="button"
              onClick={() => setSelectedLeague("ALL")}
              className={`px-3 py-1 text-xs font-medium rounded-full border transition ${
                !selectedLeague || selectedLeague === "ALL"
                  ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                  : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              ⭐ All ({predictions.length})
            </button>

            {/* Popular Leagues Filter Pills */}
            {availableLeagues.slice(0, 7).map((l) => {
              const active = selectedLeague === `league:${l.id}`;
              const flag = getCountryFlag(l.country || l.name);

              return (
                <button
                  key={l.id}
                  type="button"
                  onClick={() => setSelectedLeague(active ? "ALL" : `league:${l.id}`)}
                  className={`px-2.5 py-1 text-xs rounded-full border transition inline-flex items-center gap-1.5 ${
                    active
                      ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                      : "bg-zinc-800/60 border-zinc-700/60 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                  }`}
                >
                  <span>{flag}</span>
                  <span>{cleanLeagueName(l.name, l.country)}</span>
                  <span className="text-[10px] opacity-70 px-1 py-0.2 rounded bg-zinc-700/60">
                    {l.count}
                  </span>
                </button>
              );
            })}

            {/* More Leagues Dropdown if > 7 available */}
            {availableLeagues.length > 7 && (
              <select
                value={
                  availableLeagues.slice(7).some(
                    (l) =>
                      selectedLeague === `league:${l.id}`
                  )
                    ? selectedLeague
                    : ""
                }
                onChange={(e) => setSelectedLeague(e.target.value || "ALL")}
                className="input !py-1 !px-2 text-xs bg-zinc-800/90 border-zinc-700 text-zinc-300 !w-auto rounded-full"
                aria-label="More leagues"
              >
                <option value="">More leagues ({availableLeagues.length - 7})...</option>
                {availableLeagues.slice(7).map((l) => (
                  <option key={l.id} value={`league:${l.id}`}>
                    {cleanLeagueName(l.name, l.country)} ({l.count})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Expand / Collapse Accordion Controls */}
          <div className="flex items-center gap-2 self-end sm:self-auto shrink-0 text-xs">
            <button
              type="button"
              onClick={() => toggleAllAccordions(true)}
              className="text-zinc-400 hover:text-zinc-200 transition"
            >
              Expand All
            </button>
            <span className="text-zinc-600">·</span>
            <button
              type="button"
              onClick={() => toggleAllAccordions(false)}
              className="text-zinc-400 hover:text-zinc-200 transition"
            >
              Collapse All
            </button>
          </div>
        </div>
      </div>

      {/* Main Predictions Groups List */}
      {loading ? (
        <div className="card text-center py-16 text-zinc-400">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500 mb-3" />
          <p>Loading predictions with AI probabilities...</p>
        </div>
      ) : filteredPredictions.length === 0 ? (
        <div className="card text-center py-16 text-zinc-400 space-y-3">
          <p className="text-base text-zinc-300 font-medium">No matching predictions found</p>
          <p className="text-sm text-zinc-500">
            Try adjusting your search filter, pick filter, or time horizon.
          </p>
          <button
            type="button"
            onClick={() => {
              setSelectedLeague("ALL");
              setSelectedPick("ALL");
              setSearchQuery("");
            }}
            className="btn-secondary !py-2 !px-4 text-xs inline-block mt-2"
          >
            Reset All Filters
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {leagueGroups.map((group) => {
            const isCollapsed = !!collapsedLeagues[group.league];
            const isTopLeague = group.priority < 10;
            const flag = getCountryFlag(group.country || group.league);

            return (
              <div
                key={group.league}
                className="rounded-xl border border-zinc-700/80 bg-zinc-800/40 overflow-hidden shadow-sm"
              >
                {/* League Accordion Header with Direct Navigation Link */}
                <div className="flex items-center justify-between gap-3 px-4 py-3 bg-zinc-900/90 border-b border-zinc-700/60">
                  <button
                    type="button"
                    onClick={() => toggleGroup(group.league)}
                    className="flex items-center gap-3 min-w-0 text-left flex-1 hover:opacity-85 transition"
                  >
                    <span
                      aria-hidden
                      className={`inline-block text-zinc-400 shrink-0 text-xs transition-transform duration-200 ${
                        isCollapsed ? "rotate-0" : "rotate-90"
                      }`}
                    >
                      &#9654;
                    </span>
                    <span className="text-base">{flag}</span>
                    <span className="font-semibold text-zinc-50 truncate text-sm sm:text-base">
                      {cleanLeagueName(group.league, group.country)}
                    </span>
                    {isTopLeague && (
                      <span className="badge bg-amber-500/10 text-amber-300 border-amber-500/30 text-[10px] hidden sm:inline-block">
                        Top Flight
                      </span>
                    )}
                    <span className="badge border-zinc-700 bg-zinc-800 text-zinc-300 text-xs">
                      {group.items.length} match{group.items.length === 1 ? "" : "es"}
                    </span>
                  </button>

                  {/* Summary Breakdown & Cross-Navigation Link */}
                  <div className="flex items-center gap-3 shrink-0">
                    <div className="hidden md:flex items-center gap-1.5 text-xs text-zinc-400">
                      <span className="text-emerald-400 font-semibold">{group.homeCount}H</span>
                      <span className="text-zinc-600">·</span>
                      <span className="text-amber-400 font-semibold">{group.drawCount}D</span>
                      <span className="text-zinc-600">·</span>
                      <span className="text-rose-400 font-semibold">{group.awayCount}A</span>
                    </div>

                    <Link
                      href={`/leagues/${group.leagueId}`}
                      className="text-xs text-emerald-400 hover:text-emerald-300 hover:underline flex items-center gap-1 font-medium bg-emerald-500/10 hover:bg-emerald-500/20 px-2.5 py-1 rounded-md border border-emerald-500/20 transition"
                      title={`View full stats and historical data for ${group.league}`}
                    >
                      <span>League Stats</span>
                      <span aria-hidden>↗</span>
                    </Link>
                  </div>
                </div>

                {/* Match Rows Table */}
                {!isCollapsed && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-xs font-semibold text-zinc-400 border-b border-zinc-700/70 bg-zinc-900/30 uppercase tracking-wider">
                          <th className="px-4 py-3 min-w-[220px]">Match</th>
                          <th className="px-4 py-3 whitespace-nowrap min-w-[120px]">Date / Time</th>
                          <th className="px-4 py-3 min-w-[110px]">Predicted Pick</th>
                          <th className="px-4 py-3 min-w-[180px]">Model Probability</th>
                          <th className="px-4 py-3 min-w-[90px]">Actual</th>
                          <th className="px-4 py-3 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-700/40">
                        {group.items.map((p) => {
                          const win = getWinningProb(p);
                          const probs = p.probabilities;
                          const hProb = probs?.H ? Math.round(probs.H * 100) : null;
                          const dProb = probs?.D ? Math.round(probs.D * 100) : null;
                          const aProb = probs?.A ? Math.round(probs.A * 100) : null;

                          return (
                            <Fragment key={p.id}>
                            <tr className="hover:bg-zinc-800/30 transition group [&>td]:align-top">
                              {/* Match Teams */}
                              <td className="px-4 py-3">
                                <Link
                                  href={`/predictions/${p.id}`}
                                  className="font-medium text-zinc-100 hover:text-emerald-400 transition block"
                                >
                                  <span>{p.home_team}</span>
                                  <span className="text-zinc-500 text-xs px-1.5 font-normal">vs</span>
                                  <span>{p.away_team}</span>
                                </Link>
                                {p.markets.length > 1 && <button type="button"
                                  aria-expanded={expandedMatches.has(p.id)} aria-controls={`markets-${p.id}`}
                                  onClick={() => setExpandedMatches(previous => { const next = new Set(previous); if (next.has(p.id)) next.delete(p.id); else next.add(p.id); return next; })}
                                  className="mt-2 text-xs text-emerald-400 hover:text-emerald-300">
                                  {expandedMatches.has(p.id) ? "Hide" : "View"} {p.markets.length} market predictions {expandedMatches.has(p.id) ? "▴" : "▾"}
                                </button>}
                              </td>

                              {/* Date & Schedule */}
                              <td className="px-4 py-3 whitespace-nowrap">
                                <div className="text-zinc-200 text-xs font-medium">
                                  {formatMatchKickoff(p.match_date)}
                                </div>
                                <div className="text-zinc-500 text-[11px]">
                                  {formatDate(p.match_date)}
                                </div>
                              </td>

                              {/* Pick Outcome Badge */}<td className="px-4 py-3"><span className="text-xs font-semibold text-emerald-300">{marketPick(p.market_type, p.predicted_result)}</span><p className="text-[10px] text-zinc-500">{marketName(p.market_type)}</p></td>

                              {/* Inline Probability Distribution Bar */}<td className="px-4 py-3"><MarketProbabilities probabilities={p.probabilities}/></td>

                              {/* Actual Result */}
                              <td className="px-4 py-3 whitespace-nowrap">
                                {p.actual_result ? (
                                  <div className="flex items-center gap-1.5">
                                    <span className={`font-semibold ${resultColor(p.actual_result)}`}>
                                      {p.actual_result}
                                    </span>
                                    {p.is_correct === true && (
                                      <span className="text-emerald-400 text-xs font-bold" title="Correct">
                                        ✓
                                      </span>
                                    )}
                                    {p.is_correct === false && (
                                      <span className="text-rose-400 text-xs font-bold" title="Incorrect">
                                        ✗
                                      </span>
                                    )}
                                  </div>
                                ) : (
                                  <span className="badge bg-zinc-700/50 text-zinc-400 border-zinc-600/50 text-[11px]">
                                    Upcoming
                                  </span>
                                )}
                              </td>

                              {/* Action Link */}
                              <td className="px-4 py-3 text-right whitespace-nowrap">
                                <div className="flex items-center justify-end gap-2">
                                  <Link
                                    href={`/live?matchId=${p.id}`}
                                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/15 text-rose-300 border border-rose-500/30 hover:bg-rose-500/25 transition"
                                    title="Watch Live Match & Commentary"
                                  >
                                    <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                                    <span>Live</span>
                                  </Link>
                                  <Link
                                    href={`/predictions/${p.id}`}
                                    className="text-xs text-zinc-400 group-hover:text-emerald-400 group-hover:underline transition font-medium"
                                  >
                                    Details →
                                  </Link>
                                </div>
                              </td>
                            </tr>
                            {p.markets.length > 1 && <tr id={`markets-${p.id}`} hidden={!expandedMatches.has(p.id)}>
                              <td colSpan={6} className="bg-zinc-950/40 px-4 py-4">
                                <p className="mb-3 text-xs text-zinc-400">Markets for {p.home_team} vs {p.away_team}</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                                  {p.markets.map(m => <div key={m.id} className="rounded-lg border border-zinc-700 bg-zinc-900 p-3">
                                    <div className="flex items-start justify-between gap-3 mb-3">
                                      <div><p className="text-xs font-semibold text-emerald-300">{marketPick(m.market_type, m.predicted_result)}</p><p className="text-[10px] text-zinc-400 mt-1">{marketName(m.market_type)}</p></div>
                                      <Link href={`/predictions/${m.id}`} aria-label={`Details for ${marketName(m.market_type)}: ${p.home_team} vs ${p.away_team}`} className="shrink-0 text-xs text-emerald-400 hover:underline">Details →</Link>
                                    </div>
                                    <MarketProbabilities probabilities={m.probabilities}/>
                                    <p className="mt-2 text-xs text-zinc-400">{m.actual_result ? `Actual: ${marketPick(m.market_type, m.actual_result)}` : "Result pending"}</p>
                                  </div>)}
                                </div>
                              </td>
                            </tr>}
                            </Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            );
          })}

          {/* Pagination Controls */}
          <div className="flex items-center justify-between pt-4 border-t border-zinc-800">
            <button
              type="button"
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={!hasPrevPage}
              className="btn-secondary !py-2 !px-4 text-xs disabled:opacity-30"
            >
              ← Previous
            </button>
            <span className="text-xs text-zinc-400">
              Page {page}
              {total > 0 && (
                <span className="text-zinc-500">
                  {" "}
                  · Matches {(page - 1) * perPage + 1}–{Math.min(page * perPage, total)} of {total} · {filteredPredictions.length} match filters
                </span>
              )}
            </span>
            <button
              type="button"
              onClick={() => setPage(page + 1)}
              disabled={!hasNextPage}
              className="btn-secondary !py-2 !px-4 text-xs disabled:opacity-30"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PredictionsPage() {
  return (
    <DashboardLayout>
      <Suspense
        fallback={
          <div className="card text-center py-16 text-zinc-400">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500 mb-3" />
            <p>Loading predictions...</p>
          </div>
        }
      >
        <PredictionsContent />
      </Suspense>
    </DashboardLayout>
  );
}
