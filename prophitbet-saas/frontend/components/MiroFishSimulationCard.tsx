"use client";

import { useState, useEffect } from "react";
import { mirofishApi } from "@/lib/api";
import { cn, resultColor } from "@/lib/utils";
import MiroFishSwarmAnimation from "@/components/MiroFishSwarmAnimation";
import MathematicalAnalysisCard, { MathematicalAnalysisData } from "@/components/MathematicalAnalysisCard";

type ProbabilityMap = {
  H?: number;
  D?: number;
  A?: number;
  [key: string]: number | undefined;
};

type MiroFishSimulation = {
  id: string;
  prediction_id: string;
  swarm_predicted_result: string;
  swarm_probabilities: ProbabilityMap;
  ensemble_predicted_result: string;
  ensemble_probabilities: ProbabilityMap;
  confidence_score: number;
  consensus_level: string;
  simulation_report: {
    executive_summary?: string;
    tactical_clash?: string;
    simulated_scenarios?: { title: string; probability: number; description: string }[];
    agent_debates?: Record<string, { persona: string; argument: string; lean: string }>;
    key_risks?: string[];
    mathematical_analysis?: MathematicalAnalysisData;
  };
  created_at: string;
};

interface MiroFishSimulationCardProps {
  predictionId: string;
  token?: string;
  mlProbabilities?: ProbabilityMap | null;
  mlPredictedResult?: string;
  homeTeam?: string;
  awayTeam?: string;
  league?: string;
}

export default function MiroFishSimulationCard({
  predictionId,
  token,
  mlProbabilities,
  mlPredictedResult,
  homeTeam,
  awayTeam,
  league,
}: MiroFishSimulationCardProps) {
  const [simulation, setSimulation] = useState<MiroFishSimulation | null>(null);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>("tactical_strategist");
  const [showSwarmArena, setShowSwarmArena] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!predictionId) return;
    setInitialLoading(true);
    mirofishApi
      .getSimulation(predictionId, token)
      .then((data) => {
        if (data) setSimulation(data);
      })
      .catch(() => {})
      .finally(() => setInitialLoading(false));
  }, [predictionId, token]);

  const handleSimulate = async (force: boolean = false) => {
    setLoading(true);
    setError("");
    try {
      const res = await mirofishApi.simulate(predictionId, force, token);
      setSimulation(res);
    } catch (err: any) {
      setError(err?.message || "MiroFish simulation failed");
    } finally {
      setLoading(false);
    }
  };

  const consensusBadge = (level: string) => {
    switch (level) {
      case "STRONG_CONSENSUS":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            ⭐ High Conviction Consensus
          </span>
        );
      case "UPSET_ALERT":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-300 border border-amber-500/30">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
            ⚠️ Upset / Value Trap Alert
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-zinc-700/50 text-zinc-300 border border-zinc-600/50">
            ⚖️ Moderate Agreement
          </span>
        );
    }
  };

  const formatProb = (val?: number) => {
    if (typeof val !== "number") return "0%";
    return `${Math.round(val * 100)}%`;
  };

  const agentKeys = simulation?.simulation_report?.agent_debates
    ? Object.keys(simulation.simulation_report.agent_debates)
    : [];

  return (
    <div className="card !p-6 border-zinc-700/80 bg-gradient-to-b from-zinc-850 to-zinc-900 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-md text-xl">
            🐝
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-zinc-50 tracking-tight">
                MiroFish Swarm Intelligence
              </h2>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-teal-500/20 text-teal-300 border border-teal-500/30">
                Multi-Agent Simulation
              </span>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Powered by OASIS multi-agent sandbox • Synthesizes collective emergent forecasts
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {simulation && consensusBadge(simulation.consensus_level)}
          <button
            type="button"
            onClick={() => setShowSwarmArena(!showSwarmArena)}
            className={cn(
              "px-3 py-2 rounded-lg text-xs font-semibold border transition-all flex items-center gap-1.5",
              showSwarmArena
                ? "bg-purple-600 text-white border-purple-400 shadow-md shadow-purple-600/30"
                : "bg-zinc-800 hover:bg-zinc-750 text-purple-300 border-purple-500/30 hover:border-purple-500/50"
            )}
          >
            <span>🌌</span>
            <span>{showSwarmArena ? "Hide Swarm Arena" : "Live Swarm Arena"}</span>
          </button>
          <button
            type="button"
            onClick={() => handleSimulate(Boolean(simulation))}
            disabled={loading}
            className="btn-primary !py-2 !px-4 text-xs inline-flex items-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Simulating Swarm...</span>
              </>
            ) : simulation ? (
              <>
                <span>🔄</span>
                <span>Re-run Swarm</span>
              </>
            ) : (
              <>
                <span>⚡</span>
                <span>Run Swarm Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs">
          {error}
        </div>
      )}

      {/* Live Interactive Swarm Arena */}
      {showSwarmArena && (
        <div className="p-4 rounded-2xl bg-zinc-950/90 border border-purple-500/40 shadow-2xl space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-zinc-800">
            <div className="flex items-center gap-2">
              <span className="text-sm">🌌</span>
              <span className="text-xs font-bold text-purple-300 uppercase tracking-wider">
                Live Swarm Neural Arena
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20">
                60 FPS Boids Flocking + Bayesian Outcome Attractors
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowSwarmArena(false)}
              className="text-xs text-zinc-400 hover:text-zinc-200 px-2.5 py-1 rounded bg-zinc-800/80 hover:bg-zinc-750 transition-colors"
            >
              ✕ Close Arena
            </button>
          </div>

          <MiroFishSwarmAnimation
            match={{
              homeTeam: homeTeam || "Home Team",
              awayTeam: awayTeam || "Away Team",
              league: league || "Match League",
              probH: simulation?.swarm_probabilities?.H ?? (mlProbabilities?.H ?? 0.45),
              probD: simulation?.swarm_probabilities?.D ?? (mlProbabilities?.D ?? 0.28),
              probA: simulation?.swarm_probabilities?.A ?? (mlProbabilities?.A ?? 0.27),
              consensusLevel: simulation?.consensus_level || "STRONG_CONSENSUS",
            }}
          />
        </div>
      )}

      {initialLoading ? (
        <div className="py-10 text-center text-zinc-500 text-sm flex items-center justify-center gap-2">
          <div className="w-4 h-4 border-2 border-zinc-600 border-t-emerald-500 rounded-full animate-spin" />
          <span>Loading MiroFish multi-agent intelligence...</span>
        </div>
      ) : !simulation ? (
        /* Empty State */
        <div className="py-8 text-center space-y-4">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-800 text-2xl text-zinc-400">
            🐟
          </div>
          <div className="max-w-md mx-auto">
            <h3 className="text-sm font-semibold text-zinc-200">
              No Simulation Recorded Yet
            </h3>
            <p className="text-xs text-zinc-400 mt-1">
              Trigger MiroFish to launch 4 autonomous soccer agents (Tactical, Sharp, Morale, and Dynamics)
              into a debate round to pressure-test this fixture and fuse with the statistical ML model.
            </p>
          </div>
          <button
            type="button"
            onClick={() => handleSimulate(false)}
            disabled={loading}
            className="btn-primary !py-2 !px-5 text-xs inline-flex items-center gap-2"
          >
            {loading ? "Simulating Swarm..." : "Launch Swarm Simulation"}
          </button>
        </div>
      ) : (
        /* Active Simulation Content */
        <div className="space-y-6">
          {/* 3-Way Probability Comparison */}
          <div className="grid md:grid-cols-3 gap-4">
            {/* 1. ProphitBet ML */}
            <div className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-750 flex flex-col justify-between">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  📊 Statistical ML
                </span>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${resultColor(mlPredictedResult || "H")}`}>
                  {mlPredictedResult || "Pick"}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-zinc-300">
                  <span>Home: <strong className="text-emerald-400">{formatProb(mlProbabilities?.H)}</strong></span>
                  <span>Draw: <strong className="text-amber-400">{formatProb(mlProbabilities?.D)}</strong></span>
                  <span>Away: <strong className="text-rose-400">{formatProb(mlProbabilities?.A)}</strong></span>
                </div>
                <div className="w-full bg-zinc-900 rounded-full h-2 flex overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${(mlProbabilities?.H || 0.33) * 100}%` }} />
                  <div className="bg-amber-500 h-full" style={{ width: `${(mlProbabilities?.D || 0.33) * 100}%` }} />
                  <div className="bg-rose-500 h-full" style={{ width: `${(mlProbabilities?.A || 0.34) * 100}%` }} />
                </div>
              </div>
              <span className="text-[10px] text-zinc-500 mt-3 block">
                Historical CSV training data & Poisson regression
              </span>
            </div>

            {/* 2. MiroFish Swarm */}
            <div className="p-4 rounded-xl bg-teal-950/20 border border-teal-700/40 flex flex-col justify-between">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-teal-300 uppercase tracking-wider">
                  🐝 MiroFish Swarm
                </span>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${resultColor(simulation.swarm_predicted_result)}`}>
                  {simulation.swarm_predicted_result}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-zinc-300">
                  <span>Home: <strong className="text-emerald-400">{formatProb(simulation.swarm_probabilities?.H)}</strong></span>
                  <span>Draw: <strong className="text-amber-400">{formatProb(simulation.swarm_probabilities?.D)}</strong></span>
                  <span>Away: <strong className="text-rose-400">{formatProb(simulation.swarm_probabilities?.A)}</strong></span>
                </div>
                <div className="w-full bg-zinc-900 rounded-full h-2 flex overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${(simulation.swarm_probabilities?.H || 0.33) * 100}%` }} />
                  <div className="bg-amber-500 h-full" style={{ width: `${(simulation.swarm_probabilities?.D || 0.33) * 100}%` }} />
                  <div className="bg-rose-500 h-full" style={{ width: `${(simulation.swarm_probabilities?.A || 0.34) * 100}%` }} />
                </div>
              </div>
              <span className="text-[10px] text-teal-400/80 mt-3 block">
                1,200 simulated agent rounds • Confidence {Math.round(simulation.confidence_score * 100)}%
              </span>
            </div>

            {/* 3. Bayesian Ensemble (The Best Prediction) */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-950/30 to-zinc-850 border-2 border-emerald-500/50 flex flex-col justify-between shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1">
                  ⚡ Hybrid Ensemble
                </span>
                <span className={`text-sm font-extrabold px-2.5 py-0.5 rounded shadow ${resultColor(simulation.ensemble_predicted_result)}`}>
                  {simulation.ensemble_predicted_result}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-zinc-200">
                  <span>Home: <strong className="text-emerald-400 font-bold">{formatProb(simulation.ensemble_probabilities?.H)}</strong></span>
                  <span>Draw: <strong className="text-amber-400 font-bold">{formatProb(simulation.ensemble_probabilities?.D)}</strong></span>
                  <span>Away: <strong className="text-rose-400 font-bold">{formatProb(simulation.ensemble_probabilities?.A)}</strong></span>
                </div>
                <div className="w-full bg-zinc-900 rounded-full h-2.5 flex overflow-hidden ring-1 ring-emerald-500/30">
                  <div className="bg-emerald-500 h-full" style={{ width: `${(simulation.ensemble_probabilities?.H || 0.33) * 100}%` }} />
                  <div className="bg-amber-500 h-full" style={{ width: `${(simulation.ensemble_probabilities?.D || 0.33) * 100}%` }} />
                  <div className="bg-rose-500 h-full" style={{ width: `${(simulation.ensemble_probabilities?.A || 0.34) * 100}%` }} />
                </div>
              </div>
              <span className="text-[10px] text-emerald-400 font-medium mt-3 block">
                50% ML + 50% Swarm Weighted Bayesian Fusion
              </span>
            </div>
          </div>

          {/* Real Mathematical Derivation Card (Dixon-Coles + Value/Kelly + Empirical Stats) */}
          {simulation.simulation_report?.mathematical_analysis && (
            <MathematicalAnalysisCard
              analysis={simulation.simulation_report.mathematical_analysis}
              homeTeam={homeTeam || "Home Team"}
              awayTeam={awayTeam || "Away Team"}
            />
          )}

          {/* Executive Summary */}
          {simulation.simulation_report?.executive_summary && (
            <div className="p-4 rounded-xl bg-zinc-800/40 border border-zinc-750">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
                <span>📋</span> Swarm Synthesis & Match Dynamics
              </h3>
              <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed">
                {simulation.simulation_report.executive_summary}
              </p>
            </div>
          )}

          {/* Multi-Agent Debate Tabs */}
          {agentKeys.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                  <span>💬</span> Autonomous Agent Debate Logs
                </h3>
                <span className="text-[11px] text-zinc-500">4 distinct perspectives</span>
              </div>

              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 border-b border-zinc-800">
                {agentKeys.map((k) => {
                  const agent = simulation.simulation_report.agent_debates![k];
                  const active = activeTab === k;
                  return (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setActiveTab(k)}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 whitespace-nowrap",
                        active
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                          : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
                      )}
                    >
                      <span>{agent.persona}</span>
                      <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${resultColor(agent.lean)}`}>
                        {agent.lean}
                      </span>
                    </button>
                  );
                })}
              </div>

              {simulation.simulation_report.agent_debates?.[activeTab] && (
                <div className="p-4 rounded-xl bg-zinc-800/60 border border-zinc-750 text-xs text-zinc-300 leading-relaxed animate-in fade-in duration-150">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-zinc-100">
                      {simulation.simulation_report.agent_debates[activeTab].persona}
                    </span>
                    <span className="text-xs text-zinc-400">
                      Recommendation: <strong className="text-emerald-400">{simulation.simulation_report.agent_debates[activeTab].lean}</strong>
                    </span>
                  </div>
                  <p>{simulation.simulation_report.agent_debates[activeTab].argument}</p>
                </div>
              )}
            </div>
          )}

          {/* Simulated Match Trajectories / Scenarios */}
          {simulation.simulation_report?.simulated_scenarios && (
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <span>🎯</span> Simulated Game Trajectories
              </h3>
              <div className="grid sm:grid-cols-3 gap-3">
                {simulation.simulation_report.simulated_scenarios.map((scen, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-zinc-800/30 border border-zinc-750/80 flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <h4 className="text-xs font-bold text-zinc-200">{scen.title}</h4>
                        <span className="text-xs font-extrabold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                          {scen.probability}%
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 leading-relaxed">
                        {scen.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key Risk Factors */}
          {simulation.simulation_report?.key_risks && simulation.simulation_report.key_risks.length > 0 && (
            <div className="p-3.5 rounded-xl bg-amber-500/5 border border-amber-500/20">
              <h4 className="text-xs font-bold text-amber-300 mb-1.5 flex items-center gap-1.5">
                <span>⚠️</span> High-Impact Variance & Risk Factors
              </h4>
              <ul className="list-disc list-inside space-y-1 text-xs text-zinc-400">
                {simulation.simulation_report.key_risks.map((risk, idx) => (
                  <li key={idx}>{risk}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
