"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { marketPick, marketName } from "@/lib/markets";
import MarketProbabilities from "@/components/MarketProbabilities";
import DashboardLayout from "@/components/DashboardLayout";
import { resultsApi, leaguesApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { getCountryFlag, cleanLeagueName, matchesLeagueFilter } from "@/lib/utils";

interface MatchResultItem {
  id: string;
  league_id: number;
  league_name: string;
  country?: string;
  home_team: string;
  away_team: string;
  match_date?: string;
  actual_result: string;
  actual_score: string | null;
  market_type?: string;
  predicted_result: string;
  probabilities?: { [key: string]: number };
  confidence_score: number;
  is_correct: boolean;
  odds: number | null;
  pnl_units: number | null;
}

interface ResultsSummary {
  by_market?: {market_type: string; total: number; correct: number; accuracy_pct: number}[];
  total_settled: number;
  correct_count: number;
  accuracy_pct: number;
  net_units: number | null;
  roi_pct: number | null;
  home_accuracy_pct: number;
  draw_accuracy_pct: number;
  away_accuracy_pct: number;
  brier_score: number | null;
  calibration_grade: string;
  per_league: Array<{
    league_id: number;
    league: string;
    country?: string;
    total: number;
    correct: number;
    accuracy_pct: number;
  }>;
}

interface LearningFeedback {
  settled_count: number;
  learning_delta: string;
  brier_score_before: number | null;
  brier_score_after: number | null;
  calibrated_features: string[];
  status: string;
  message: string;
}

const POPULAR_LEAGUES = [
  { id: "ALL", name: "All Leagues", flag: "🌍" },
  { id: "Premier-League", name: "Premier League (ENG)", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
  { id: "Championship", name: "Championship (ENG)", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
  { id: "PSL", name: "Betway Premiership (RSA)", flag: "🇿🇦" },
  { id: "La-Liga", name: "La Liga (ESP)", flag: "🇪🇸" },
  { id: "Serie-A", name: "Serie A (ITA)", flag: "🇮🇹" },
  { id: "Bundesliga", name: "Bundesliga (GER)", flag: "🇩🇪" },
  { id: "Ligue-1", name: "Ligue 1 (FRA)", flag: "🇫🇷" },
];

export default function ResultsAndAccuracyPage() {
  const { token, user } = useAuth();

  // Filter States
  const [selectedLeague, setSelectedLeague] = useState<string>("ALL");
  const [outcomeFilter, setOutcomeFilter] = useState<string>("ALL");
  const [accuracyFilter, setAccuracyFilter] = useState<string>("ALL");
  const [daysFilter, setDaysFilter] = useState<number>(90);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const perPage = 25;

  // Data States
  const [results, setResults] = useState<MatchResultItem[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [summary, setSummary] = useState<ResultsSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [summaryLoading, setSummaryLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // System Learning & Model Calibration State
  const [learningLoading, setLearningLoading] = useState<boolean>(false);
  const [learningFeedback, setLearningFeedback] = useState<LearningFeedback | null>(null);
  const [showLeagueBreakdown, setShowLeagueBreakdown] = useState<boolean>(false);

  // 1. Fetch Summary Metrics
  const fetchSummary = async () => {
    try {
      setSummaryLoading(true);
      const data = await resultsApi.summary(daysFilter, token);
      setSummary(data);
    } catch (err: any) {
      console.error("Failed to load results summary:", err);
    } finally {
      setSummaryLoading(false);
    }
  };

  // 2. Fetch Settled Results List
  const fetchResults = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await resultsApi.list(
        {
          league: selectedLeague !== "ALL" ? selectedLeague : undefined,
          outcome: outcomeFilter,
          accuracy: accuracyFilter,
          days: daysFilter,
          page: currentPage,
          per_page: perPage,
        },
        token
      );
      setResults(data.results || []);
      setTotalCount(data.total || 0);
    } catch (err: any) {
      console.error("Failed to load match results:", err);
      setError(err?.message || "Failed to load match results. Please check connection.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, [daysFilter, token]);

  useEffect(() => {
    fetchResults();
  }, [selectedLeague, outcomeFilter, accuracyFilter, daysFilter, currentPage, token]);

  // Trigger Continuous Learning Engine
  const handleTriggerLearning = async () => {
    try {
      setLearningLoading(true);
      const resp = await resultsApi.triggerLearning(token);
      setLearningFeedback(resp);
      // Refresh results and summary after system learning
      await Promise.all([fetchResults(), fetchSummary()]);
    } catch (err: any) {
      console.error("System learning failed:", err);
      alert("System learning encountered an issue: " + (err?.message || "Internal error"));
    } finally {
      setLearningLoading(false);
    }
  };

  // Client-side search & strict filter reassurance
  const displayedResults = useMemo(() => {
    return results.filter((item) => {
      // Strict league matching
      if (selectedLeague !== "ALL") {
        if (!matchesLeagueFilter(item, selectedLeague)) {
          return false;
        }
      }
      // Text search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchTitle = `${item.home_team} vs ${item.away_team}`.toLowerCase();
        const league = (item.league_name || "").toLowerCase();
        if (!matchTitle.includes(q) && !league.includes(q)) {
          return false;
        }
      }
      return true;
    });
  }, [results, selectedLeague, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(totalCount / perPage));

  return (
    <DashboardLayout>
      <div className="space-y-6 pb-16">
        {/* Header Title & System Learning Action */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-zinc-800/80 pb-5">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="flex h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
              <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-zinc-100">
                Match Results & Prediction Audit
              </h1>
              <span className="px-2 py-0.5 text-xs font-bold uppercase rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Verified Factual
              </span>
            </div>
            <p className="mt-1 text-sm text-zinc-400">
              Complete historical ledger comparing verified final match scores against AI model predictions, with continuous feedback learning.
            </p>
          </div>

          {user?.is_admin && <div className="flex items-center gap-3">
            <button
              id="btn-trigger-learning"
              onClick={handleTriggerLearning}
              disabled={learningLoading}
              className="group relative inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-emerald-600 px-4 py-2.5 text-xs sm:text-sm font-bold text-white shadow-lg shadow-purple-900/30 hover:brightness-110 active:scale-95 transition-all disabled:opacity-50"
            >
              <span className="text-base">{learningLoading ? "⚡" : "🧠"}</span>
              <span>{learningLoading ? "Calibrating Neural Weights..." : "Learn From All Results & Calibrate"}</span>
            </button>
          </div>}
        </div>

        {/* System Learning Feedback Banner */}
        {learningFeedback && (
          <div className="rounded-2xl border border-purple-500/30 bg-purple-950/20 p-5 backdrop-blur-md transition-all animate-fadeIn">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-purple-600/30 text-purple-300 border border-purple-500/40 text-lg">
                  ✨
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-purple-200 text-sm">System Learning Completed</h3>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono font-bold">
                      {learningFeedback.learning_delta}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-0.5">{learningFeedback.message}</p>
                </div>
              </div>

              <div className="flex items-center gap-4 bg-zinc-900/80 px-4 py-2 rounded-xl border border-zinc-800 text-xs">
                <div>
                  <span className="text-zinc-500 block text-[10px] uppercase font-mono">Brier Error Before</span>
                  <span className="font-mono font-bold text-zinc-300">{learningFeedback.brier_score_before?.toFixed(3) ?? "N/A"}</span>
                </div>
                <div className="text-zinc-600">→</div>
                <div>
                  <span className="text-zinc-500 block text-[10px] uppercase font-mono">Brier After</span>
                  <span className="font-mono font-bold text-emerald-400">{learningFeedback.brier_score_after?.toFixed(3) ?? "N/A"}</span>
                </div>
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-purple-500/20 flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-medium text-purple-300/80">Recalibrated Model Weights:</span>
              {learningFeedback.calibrated_features.map((feat, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 rounded bg-zinc-900/90 text-zinc-300 border border-zinc-800 text-[10px] font-mono"
                >
                  ✓ {feat}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Executive KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-4 backdrop-blur-sm">
            <span className="text-xs font-mono uppercase text-zinc-400">Total Settled</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-black text-zinc-100">
                {summaryLoading ? "..." : summary?.total_settled ?? 0}
              </span>
              <span className="text-xs text-zinc-500">matches</span>
            </div>
            <div className="mt-2 text-[11px] text-zinc-400">
              Verified outcomes & final scores
            </div>
          </div>

          <div className="rounded-2xl border border-emerald-500/20 bg-emerald-950/10 p-4 backdrop-blur-sm">
            <span className="text-xs font-mono uppercase text-emerald-400/90">Prediction Accuracy</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-black text-emerald-400">
                {summaryLoading ? "..." : `${summary?.accuracy_pct.toFixed(1)}%`}
              </span>
              <span className="text-xs text-emerald-500/80">
                ({summary?.correct_count ?? 0} hits)
              </span>
            </div>
            <div className="mt-2 text-[11px] text-zinc-400">
              Strict 1X2 market settlement
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/60 p-4 backdrop-blur-sm">
            <span className="text-xs font-mono uppercase text-zinc-400">Recorded Net Units</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span
                className={`text-2xl sm:text-3xl font-black ${
                  (summary?.net_units ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {summaryLoading
                  ? "..."
                  : summary?.net_units == null
                    ? "N/A"
                    : `${summary.net_units > 0 ? "+" : ""}${summary.net_units.toFixed(1)}u`}
              </span>
              <span className="text-xs text-zinc-500">
                {summary?.roi_pct == null ? "(odds unavailable)" : `(${summary.roi_pct > 0 ? "+" : ""}${summary.roi_pct.toFixed(1)}% ROI)`}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-zinc-400">
              Calculated only from recorded bet odds
            </div>
          </div>

          <div className="rounded-2xl border border-purple-500/20 bg-purple-950/10 p-4 backdrop-blur-sm">
            <span className="text-xs font-mono uppercase text-purple-400/90">1X2 Calibration & Brier</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-2xl sm:text-3xl font-black text-purple-300">
                {summaryLoading ? "..." : summary?.brier_score?.toFixed(3) ?? "N/A"}
              </span>
              <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {summary?.calibration_grade ?? "B+"}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-zinc-400">
              Probability sharpness metric
            </div>
          </div>
        </div>

        {/* League Breakdown Toggle Bar */}
        {!!summary?.by_market?.length && <section className="card"><h2 className="mb-3 font-semibold">Accuracy by prediction market</h2><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{summary.by_market.map(m => <div key={m.market_type} className="rounded-lg bg-zinc-900 p-3 text-sm"><p>{marketName(m.market_type)}</p><p className="mt-1 text-zinc-400">{m.correct}/{m.total} correct · {m.accuracy_pct}%</p></div>)}</div><p className="mt-3 text-xs text-zinc-500">Different markets have different baselines. These figures are historical results, not guarantees.</p></section>}
        {summary?.per_league && summary.per_league.length > 0 && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-zinc-200">Per-League Accuracy Audit</span>
                <span className="text-xs text-zinc-500 font-mono">({summary.per_league.length} leagues tracked)</span>
              </div>
              <button
                onClick={() => setShowLeagueBreakdown(!showLeagueBreakdown)}
                className="text-xs text-emerald-400 hover:text-emerald-300 font-medium transition-colors"
              >
                {showLeagueBreakdown ? "Hide Breakdown ▲" : "Show League Breakdown ▼"}
              </button>
            </div>

            {showLeagueBreakdown && (
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-3 border-t border-zinc-800">
                {summary.per_league.map((pl, idx) => (
                  <div
                    key={idx}
                    onClick={() => {
                      setSelectedLeague(`league:${pl.league_id}`);
                      setCurrentPage(1);
                    }}
                    className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer transition-all ${
                      selectedLeague === `league:${pl.league_id}`
                        ? "border-emerald-500/60 bg-emerald-950/20"
                        : "border-zinc-800 bg-zinc-900/80 hover:border-zinc-700"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span>{getCountryFlag(pl.league)}</span>
                      <div>
                        <div className="text-xs font-semibold text-zinc-200">{cleanLeagueName(pl.league)}</div>
                        <div className="text-[10px] text-zinc-500">{pl.total} matches settled</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <span
                        className={`text-xs font-bold font-mono ${
                          pl.accuracy_pct >= 40
                            ? "text-emerald-400"
                            : pl.accuracy_pct >= 30
                            ? "text-amber-400"
                            : "text-zinc-400"
                        }`}
                      >
                        {pl.accuracy_pct.toFixed(1)}%
                      </span>
                      <div className="text-[10px] text-zinc-500 font-mono">
                        {pl.correct}/{pl.total} hits
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Filter Controls & Search */}
        <div className="space-y-3 rounded-2xl border border-zinc-800/80 bg-zinc-900/80 p-4">
          {/* Quick League Filter Pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
            {POPULAR_LEAGUES.map((lg) => (
              <button
                key={lg.id}
                onClick={() => {
                  setSelectedLeague(lg.id);
                  setCurrentPage(1);
                }}
                className={`flex shrink-0 items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition-all ${
                  selectedLeague === lg.id
                    ? "bg-emerald-600 text-white shadow-md shadow-emerald-900/30"
                    : "bg-zinc-800/80 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                }`}
              >
                <span>{lg.flag}</span>
                <span>{lg.name}</span>
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2">
            {/* Search Input */}
            <div>
              <label className="text-[11px] font-mono text-zinc-400 uppercase block mb-1">Search Teams</label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="e.g. Arsenal, Chiefs, Real..."
                className="w-full rounded-xl border border-zinc-700/80 bg-zinc-950/80 px-3 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none"
              />
            </div>

            {/* Accuracy Filter */}
            <div>
              <label className="text-[11px] font-mono text-zinc-400 uppercase block mb-1">Prediction Accuracy</label>
              <select
                value={accuracyFilter}
                onChange={(e) => {
                  setAccuracyFilter(e.target.value);
                  setCurrentPage(1);
                }}
                className="w-full rounded-xl border border-zinc-700/80 bg-zinc-950/80 px-3 py-2 text-xs text-zinc-100 focus:border-emerald-500 focus:outline-none"
              >
                <option value="ALL">All Outcomes (Hits & Misses)</option>
                <option value="CORRECT">✓ Hits Only (Model Won)</option>
                <option value="INCORRECT">✗ Misses Only (Model Lost)</option>
              </select>
            </div>

            {/* Outcome Filter */}
            <div>
              <label className="text-[11px] font-mono text-zinc-400 uppercase block mb-1">Match Result</label>
              <select
                value={outcomeFilter}
                onChange={(e) => {
                  setOutcomeFilter(e.target.value);
                  setCurrentPage(1);
                }}
                className="w-full rounded-xl border border-zinc-700/80 bg-zinc-950/80 px-3 py-2 text-xs text-zinc-100 focus:border-emerald-500 focus:outline-none"
              >
                <option value="ALL">All Final Results (1X2)</option>
                <option value="H">Home Wins (H)</option>
                <option value="D">Draws (D)</option>
                <option value="A">Away Wins (A)</option>
              </select>
            </div>

            {/* Lookback Window */}
            <div>
              <label className="text-[11px] font-mono text-zinc-400 uppercase block mb-1">Lookback Window</label>
              <select
                value={daysFilter}
                onChange={(e) => {
                  setDaysFilter(Number(e.target.value));
                  setCurrentPage(1);
                }}
                className="w-full rounded-xl border border-zinc-700/80 bg-zinc-950/80 px-3 py-2 text-xs text-zinc-100 focus:border-emerald-500 focus:outline-none"
              >
                <option value={14}>Last 14 Days</option>
                <option value={30}>Last 30 Days</option>
                <option value={90}>Last 90 Days</option>
                <option value={180}>Last 6 Months</option>
                <option value={365}>Full Season (365 Days)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Results & Audit Table */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 overflow-hidden shadow-xl backdrop-blur-sm">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-zinc-200">Historical Match Results Ledger</span>
              <span className="text-xs text-zinc-400 font-mono">
                (Showing {displayedResults.length} of {totalCount} records)
              </span>
            </div>
            {selectedLeague !== "ALL" && (
              <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 font-medium">
                Filtered to: {cleanLeagueName(selectedLeague)}
              </span>
            )}
          </div>

          {loading ? (
            <div className="py-16 text-center">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-zinc-700 border-t-emerald-500" />
              <p className="mt-3 text-xs text-zinc-400 font-mono">Loading factual match results...</p>
            </div>
          ) : error ? (
            <div className="p-8 text-center text-rose-400 text-sm">
              <p>{error}</p>
              <button
                onClick={fetchResults}
                className="mt-3 px-4 py-1.5 rounded-xl bg-zinc-800 text-xs text-zinc-200 hover:bg-zinc-700"
              >
                Retry
              </button>
            </div>
          ) : displayedResults.length === 0 ? (
            <div className="py-16 text-center">
              <div className="text-3xl">⚽</div>
              <h3 className="mt-2 text-sm font-semibold text-zinc-300">No settled match results found</h3>
              <p className="mt-1 text-xs text-zinc-500">
                {selectedLeague !== "ALL"
                  ? `No verified match results currently recorded under ${cleanLeagueName(selectedLeague)}.`
                  : "Adjust your filters or lookback window to view historical results."}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-950/60 text-[10px] font-mono uppercase text-zinc-400">
                    <th className="py-3 px-4">Date / Kickoff</th>
                    <th className="py-3 px-4">League & Match</th>
                    <th className="py-3 px-4 text-center">Factual Score</th>
                    <th className="py-3 px-4 text-center">Predicted Pick</th>
                    <th className="py-3 px-4">Model Probabilities</th>
                    <th className="py-3 px-4 text-center">Audit Result</th>
                    <th className="py-3 px-4 text-right">Odds / P&L</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 font-medium text-zinc-200">
                  {displayedResults.map((item) => {
                    const matchDateFormatted = item.match_date
                      ? new Date(item.match_date).toLocaleDateString(undefined, {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })
                      : "Completed";

                    const pickLabel = marketPick(item.market_type, item.predicted_result);

                    const pHome = item.probabilities?.H ? (item.probabilities.H * 100).toFixed(0) : "—";
                    const pDraw = item.probabilities?.D ? (item.probabilities.D * 100).toFixed(0) : "—";
                    const pAway = item.probabilities?.A ? (item.probabilities.A * 100).toFixed(0) : "—";

                    return (
                      <tr
                        key={item.id}
                        className="hover:bg-zinc-800/30 transition-colors group"
                      >
                        {/* Match Date */}
                        <td className="py-3 px-4 text-zinc-400 whitespace-nowrap font-mono text-[11px]">
                          {matchDateFormatted}
                        </td>

                        {/* League & Teams */}
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                            <span>{getCountryFlag(item.league_name)}</span>
                            <span className="font-semibold text-zinc-300">
                              {cleanLeagueName(item.league_name)}
                            </span>
                          </div>
                          <div className="text-sm font-bold text-zinc-100 mt-0.5">
                            {item.home_team} <span className="text-zinc-500 font-normal">vs</span> {item.away_team}
                          </div>
                        </td>

                        {/* Factual Score */}
                        <td className="py-3 px-4 text-center whitespace-nowrap">
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-zinc-950 border border-zinc-700/80 font-mono font-black text-sm text-zinc-100 shadow-inner">
                            {item.actual_score ?? "Score unavailable"}
                          </span>
                          <span className="block text-[10px] text-zinc-500 font-mono mt-0.5">
                            Outcome: {marketPick(item.market_type, item.actual_result)}
                          </span>
                        </td>

                        {/* Predicted Pick */}
                        <td className="py-3 px-4 text-center whitespace-nowrap">
                          <span
                            className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-bold ${
                              item.predicted_result === "H"
                                ? "bg-blue-500/10 text-blue-400 border border-blue-500/30"
                                : item.predicted_result === "A"
                                ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                                : "bg-purple-500/10 text-purple-400 border border-purple-500/30"
                            }`}
                          >
                            {pickLabel}
                          </span>
                          <span className="block text-[10px] text-zinc-400 font-mono mt-0.5">
                            {item.confidence_score.toFixed(0)}% Conf
                          </span>
                        </td>

                        {/* Model Probabilities */}<td className="py-3 px-4"><MarketProbabilities probabilities={item.probabilities}/></td>

                        {/* Audit Result (Hit vs Miss) */}
                        <td className="py-3 px-4 text-center whitespace-nowrap">
                          {item.is_correct ? (
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2.5 py-1 text-xs font-bold text-emerald-400 border border-emerald-500/30">
                              ✓ HIT
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/20 px-2.5 py-1 text-xs font-bold text-rose-400 border border-rose-500/30">
                              ✗ MISS
                            </span>
                          )}
                        </td>

                        {/* Odds & PnL */}
                        <td className="py-3 px-4 text-right whitespace-nowrap font-mono">
                          <div className="text-xs font-bold text-zinc-300">
                            {item.odds == null ? "N/A" : `@${item.odds.toFixed(2)}`}
                          </div>
                          <div
                            className={`text-[11px] font-bold ${
                              (item.pnl_units ?? 0) > 0
                                ? "text-emerald-400"
                                : (item.pnl_units ?? 0) < 0
                                ? "text-rose-400"
                                : "text-zinc-400"
                            }`}
                          >
                            {item.pnl_units == null
                              ? "N/A"
                              : item.pnl_units > 0
                                ? `+${item.pnl_units.toFixed(2)}u`
                                : `${item.pnl_units.toFixed(2)}u`}
                          </div>
                        </td>

                        {/* Actions */}
                        <td className="py-3 px-4 text-right whitespace-nowrap">
                          <div className="flex items-center justify-end gap-1.5">
                            <Link
                              href={`/live?matchId=${item.id}`}
                              className="rounded-lg bg-zinc-800 px-2 py-1 text-[10px] font-semibold text-zinc-300 hover:bg-zinc-700 hover:text-white transition-colors"
                              title="Watch full commentary replay"
                            >
                              Live Watch
                            </Link>
                            <Link
                              href={`/predictions/${item.id}`}
                              className="rounded-lg bg-emerald-950/60 border border-emerald-800/40 px-2 py-1 text-[10px] font-semibold text-emerald-300 hover:bg-emerald-900/60 transition-colors"
                              title="View prediction audit details"
                            >
                              Inspect
                            </Link>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="px-5 py-3 border-t border-zinc-800 flex items-center justify-between text-xs text-zinc-400">
              <div>
                Page {currentPage} of {totalPages}
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
                >
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 rounded-lg bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
