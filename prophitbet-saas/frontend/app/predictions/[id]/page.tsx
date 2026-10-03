"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { marketName, marketPick } from "@/lib/markets";
import DashboardLayout from "@/components/DashboardLayout";
import MiroFishSimulationCard from "@/components/MiroFishSimulationCard";
import { useAuth } from "@/lib/auth";
import { predictionsApi } from "@/lib/api";
import { formatDate, formatMatchDay, formatMatchKickoff, cleanLeagueName, getCountryFlag, resultColor } from "@/lib/utils";

export default function PredictionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { token } = useAuth();
  const [pred, setPred] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token || !id) return;
    predictionsApi
      .get(id, token)
      .then(setPred)
      .catch(() => setPred(null))
      .finally(() => setLoading(false));
  }, [token, id]);

  const getPickLabel = (pick: string, homeTeam: string, awayTeam: string) => {
    if (pred?.market_type && pred.market_type !== "result") return marketPick(pred.market_type, pick);
    switch (pick?.toUpperCase()) {
      case "H":
        return `Home Win (${homeTeam})`;
      case "D":
        return "Draw (Tie)";
      case "A":
        return `Away Win (${awayTeam})`;
      default:
        return pick || "Pending";
    }
  };

  const getProbColor = (pick: string) => {
    switch (pick?.toUpperCase()) {
      case "H":
        return "from-emerald-500 to-emerald-400";
      case "D":
        return "from-amber-500 to-amber-400";
      case "A":
        return "from-blue-500 to-indigo-400";
      default:
        return "from-emerald-500 to-teal-400";
    }
  };

  const getProbBg = (pick: string) => {
    switch (pick?.toUpperCase()) {
      case "H":
        return "bg-emerald-500/10 border-emerald-500/30 text-emerald-400";
      case "D":
        return "bg-amber-500/10 border-amber-500/30 text-amber-400";
      case "A":
        return "bg-blue-500/10 border-blue-500/30 text-blue-400";
      default:
        return "bg-zinc-800 border-zinc-700 text-zinc-300";
    }
  };

  return (
    <DashboardLayout>
      {loading ? (
        <div className="flex flex-col items-center justify-center py-28 space-y-4">
          <div className="animate-spin rounded-full h-9 w-9 border-b-2 border-emerald-500" />
          <p className="text-zinc-400 text-sm">Loading match prediction analysis...</p>
        </div>
      ) : !pred ? (
        <div className="card text-center py-20 max-w-lg mx-auto space-y-4">
          <div className="w-12 h-12 mx-auto rounded-full bg-zinc-800 flex items-center justify-center text-xl text-zinc-400">
            🔍
          </div>
          <h2 className="text-xl font-bold text-zinc-100">Prediction Not Found</h2>
          <p className="text-zinc-400 text-sm">
            We couldn&apos;t find a prediction for this ID. It may have expired or belongs to another dataset.
          </p>
          <Link href="/predictions" className="btn-primary !py-2 !px-4 text-sm inline-block">
            ← Back to Predictions
          </Link>
        </div>
      ) : (
        <div className="max-w-4xl space-y-6">
          {/* Breadcrumb Navigation */}
          <nav className="flex items-center gap-2 text-xs text-zinc-400 mb-2 flex-wrap" aria-label="Breadcrumb">
            <Link href="/dashboard" className="hover:text-zinc-200 transition">
              Dashboard
            </Link>
            <span className="text-zinc-600">/</span>
            <Link href="/predictions" className="hover:text-zinc-200 transition">
              Predictions
            </Link>
            <span className="text-zinc-600">/</span>
            <Link
              href={`/predictions?league=${encodeURIComponent(pred.league_name)}&leagueId=${pred.league_id}`}
              className="hover:text-zinc-200 transition inline-flex items-center gap-1.5 text-zinc-300"
            >
              <span>{getCountryFlag(pred.country || pred.league_name)}</span>
              <span>{cleanLeagueName(pred.league_name, pred.country)}</span>
            </Link>
            <span className="text-zinc-600">/</span>
            <span className="text-emerald-400 font-medium truncate max-w-xs">
              {pred.home_team} vs {pred.away_team}
            </span>
          </nav>

          {/* Match Hero Header */}
          <div className="card bg-gradient-to-br from-zinc-800/90 via-zinc-800/50 to-zinc-900/90 border-zinc-700/80 p-6 shadow-surface">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
              <div>
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="badge bg-zinc-800 text-zinc-300 border-zinc-700 text-xs inline-flex items-center gap-1.5">
                    <span>{getCountryFlag(pred.country || pred.league_name)}</span>
                    <span>{cleanLeagueName(pred.league_name, pred.country)}</span>
                  </span>
                  <span className="text-xs text-zinc-300 font-medium">
                    {formatMatchKickoff(pred.match_date)}
                  </span>
                  <span className="text-xs text-zinc-500">
                    ({formatDate(pred.match_date)})
                  </span>
                </div>
                <h1 className="anim-page-title text-2xl sm:text-3xl font-bold tracking-tight text-white">
                  <span className="text-zinc-100">{pred.home_team}</span>
                  <span className="text-zinc-500 font-normal px-2.5 text-xl sm:text-2xl">vs</span>
                  <span className="text-zinc-100">{pred.away_team}</span>
                </h1>
              </div>

              <div className="flex items-center gap-2.5 flex-wrap">
                <Link
                  href={`/live?matchId=${pred.id}`}
                  className="btn-primary !py-2 !px-3.5 text-xs inline-flex items-center gap-1.5 shadow-sm bg-rose-600 hover:bg-rose-500 border-rose-500"
                >
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                  </span>
                  <span>Watch Live</span>
                </Link>
                <Link
                  href="/predictions"
                  className="btn-secondary !py-2 !px-3.5 text-xs inline-flex items-center gap-1.5"
                >
                  ← All Predictions
                </Link>
                <Link
                  href={`/leagues/${pred.league_id}`}
                  className="btn-secondary !py-2 !px-3.5 text-xs inline-flex items-center gap-1.5"
                >
                  Match Data ↗
                </Link>
                <Link
                  href="/fixtures"
                  className="btn-secondary !py-2 !px-3.5 text-xs inline-flex items-center gap-1.5"
                >
                  Fixtures ↗
                </Link>
              </div>
            </div>
          </div>

          {/* Key Metric Overview Cards */}
          <div className="anim-features-section grid sm:grid-cols-2 gap-4">
            <div className="anim-feature-card card border-zinc-700/80 bg-zinc-800/60 p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                  Model Prediction
                </p>
                <span className="badge bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-[10px]">
                  Algorithm Choice
                </span>
              </div>
              <div className="flex items-baseline gap-3">
                <p className={`text-3xl font-extrabold tracking-tight ${resultColor(pred.predicted_result)}`}>
                  {pred.predicted_result}
                </p>
                <span className="text-sm font-medium text-zinc-300">
                  {getPickLabel(pred.predicted_result, pred.home_team, pred.away_team)}
                </span>
              </div>
            </div>

            <div className="anim-feature-card card border-zinc-700/80 bg-zinc-800/60 p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                  Match Outcome
                </p>
                <span className={`badge text-[10px] ${
                  pred.actual_result ? "bg-zinc-800 text-zinc-300 border-zinc-700" : "bg-blue-500/10 text-blue-400 border-blue-500/30"
                }`}>
                  {pred.actual_result ? "Result Settled" : "Scheduled"}
                </span>
              </div>
              <div className="flex items-baseline gap-3">
                <p className={`text-3xl font-extrabold tracking-tight ${resultColor(pred.actual_result || "")}`}>
                  {pred.actual_result || "—"}
                </p>
                <span className="text-sm font-medium text-zinc-400">
                  {pred.actual_result
                    ? getPickLabel(pred.actual_result, pred.home_team, pred.away_team)
                    : "Match not yet played / In progress"}
                </span>
              </div>
            </div>
          </div>

          {/* Probability Breakdown Section */}
          {pred.probabilities && (
            <div className="anim-reveal-section card border-zinc-700/80 bg-zinc-800/60 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-white tracking-tight">Outcome Probability Breakdown</h2>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Model probability estimates for {marketName(pred.market_type)} — regulation time
                  </p>
                </div>
                <span className="badge bg-zinc-800 text-zinc-300 border-zinc-700 text-[11px]">
                  100% Normalized
                </span>
              </div>

              <div className="space-y-4 pt-2">
                {Object.entries(pred.probabilities).map(([key, rawVal]) => {
                  const prob = typeof rawVal === "number" ? rawVal : 0;
                  const pct = (prob * 100).toFixed(1);
                  const isPick = pred.predicted_result === key;
                  const impliedOdds = prob > 0 ? (1 / prob).toFixed(2) : "—";

                  let labelTitle = "";
                  if (key === "H") labelTitle = `Home: ${pred.home_team}`;
                  else if (key === "D") labelTitle = "Draw";
                  else if (key === "A") labelTitle = `Away: ${pred.away_team}`;
                  else labelTitle = marketPick(pred.market_type, key);

                  return (
                    <div
                      key={key}
                      className={`p-3.5 rounded-lg border transition-all ${
                        isPick
                          ? "bg-zinc-800/90 border-emerald-500/40 shadow-sm"
                          : "bg-zinc-900/40 border-zinc-800 hover:border-zinc-700"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold ${getProbBg(key)}`}>
                            {key}
                          </span>
                          <span className="text-sm font-semibold text-zinc-200">{labelTitle}</span>
                          {isPick && (
                            <span className="badge bg-emerald-500/20 text-emerald-300 border-emerald-500/30 text-[10px]">
                              ⭐ Top Pick
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-3">
                          <span className="text-xs text-zinc-400 hidden sm:inline">
                            Fair Odds: <strong className="text-zinc-200">@{impliedOdds}</strong>
                          </span>
                          <span className="text-base font-bold text-white">{pct}%</span>
                        </div>
                      </div>

                      <div className="w-full bg-zinc-800 rounded-full h-2.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${getProbColor(key)} transition-all duration-500`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* MiroFish Swarm Intelligence Multi-Agent Simulation */}
          {(!pred.market_type || pred.market_type === "result") && <MiroFishSimulationCard
            predictionId={id}
            token={token || undefined}
            mlProbabilities={pred.probabilities}
            mlPredictedResult={pred.predicted_result}
            homeTeam={pred.home_team}
            awayTeam={pred.away_team}
            league={pred.league_name || pred.league?.name}
          />}
          {pred.market_type && pred.market_type !== "result" && <p className="text-xs text-zinc-400">MiroFish match-winner simulation is not applied to this market. These are the dedicated market model’s predictions.</p>}

          {/* Settled Accuracy Verification */}
          {pred.is_correct !== null && pred.actual_result && (
            <div className="anim-reveal-section">
              <div
                className={`card p-5 border flex items-center gap-3 ${
                  pred.is_correct
                    ? "bg-emerald-950/20 border-emerald-500/40 text-emerald-300"
                    : "bg-red-950/20 border-red-500/40 text-red-300"
                }`}
              >
                <span className="text-2xl">{pred.is_correct ? "✅" : "❌"}</span>
                <div>
                  <h3 className="font-bold text-sm">
                    {pred.is_correct ? "Accurate Match Prediction" : "Prediction Diverged from Actual"}
                  </h3>
                  <p className="text-xs opacity-80 mt-0.5">
                    Model predicted <strong className="underline">{pred.predicted_result}</strong> and actual outcome was{" "}
                    <strong className="underline">{pred.actual_result}</strong>.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </DashboardLayout>
  );
}
