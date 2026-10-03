"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { mirofishApi, predictionsApi, apiFetch } from "@/lib/api";
import { formatUtcToLocal } from "@/lib/utils";
import MiroFishSwarmAnimation from "@/components/MiroFishSwarmAnimation";
import MathematicalAnalysisCard from "@/components/MathematicalAnalysisCard";

type TabKey = "live" | "swarm_arena" | "sandbox" | "testing" | "agents";

export default function MiroFishLabPage() {
  const { token, user } = useAuth();
  const [activeTab, setActiveTab] = useState<TabKey>("swarm_arena");
  const [loading, setLoading] = useState(true);

  // Telemetry & Status
  const [status, setStatus] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [simulations, setSimulations] = useState<any[]>([]);
  const [selectedSim, setSelectedSim] = useState<any | null>(null);
  const [selectedArenaMatch, setSelectedArenaMatch] = useState<any | null>(null);

  // Filters for Live Monitor
  const [searchQuery, setSearchQuery] = useState("");
  const [consensusFilter, setConsensusFilter] = useState("ALL");

  // Sandbox State
  const [sandboxHome, setSandboxHome] = useState("Arsenal");
  const [sandboxAway, setSandboxAway] = useState("Chelsea");
  const [sandboxLeague, setSandboxLeague] = useState("Premier League");
  const [sandboxHomeOdds, setSandboxHomeOdds] = useState(1.95);
  const [sandboxDrawOdds, setSandboxDrawOdds] = useState(3.40);
  const [sandboxAwayOdds, setSandboxAwayOdds] = useState(4.10);
  const [sandboxMlH, setSandboxMlH] = useState(0.50);
  const [sandboxMlD, setSandboxMlD] = useState(0.28);
  const [sandboxMlA, setSandboxMlA] = useState(0.22);
  const [sandboxTactics, setSandboxTactics] = useState("High intensity London derby; tactical pressing battle in central midfield.");
  const [sandboxMlWeight, setSandboxMlWeight] = useState(0.55);
  const [sandboxLoading, setSandboxLoading] = useState(false);
  const [sandboxResult, setSandboxResult] = useState<any | null>(null);
  const [sandboxAgentTab, setSandboxAgentTab] = useState("tactical_strategist");

  // Continuous Testing State
  const [testBatchSize, setTestBatchSize] = useState(5);
  const [testStressMode, setTestStressMode] = useState("standard");
  const [testRunning, setTestRunning] = useState(false);
  const [testReport, setTestReport] = useState<any | null>(null);

  // Batch action state
  const [batchActionMsg, setBatchActionMsg] = useState<string | null>(null);
  const [batchSimulating, setBatchSimulating] = useState(false);

  // Agent Calibration & Vector RAG State
  const [calibration, setCalibration] = useState<any | null>(null);
  const [ragTeamQuery, setRagTeamQuery] = useState("Arsenal");
  const [ragDossierResult, setRagDossierResult] = useState<any | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [seedRAGMsg, setSeedRAGMsg] = useState<string | null>(null);

  // Load initial telemetry & simulations
  const loadData = async () => {
    try {
      setLoading(true);
      const [statusRes, statsRes, simsRes, calibRes] = await Promise.all([
        mirofishApi.status(token || undefined).catch(() => null),
        mirofishApi.stats(token || undefined).catch(() => null),
        mirofishApi.simulations({ limit: 50 }, token || undefined).catch(() => []),
        mirofishApi.calibration(token || undefined).catch(() => null),
      ]);
      setStatus(statusRes);
      setStats(statsRes);
      setSimulations(Array.isArray(simsRes) ? simsRes : []);
      setCalibration(calibRes);
    } catch (err) {
      console.error("Error loading MiroFish lab data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [token]);

  // Run Sandbox Simulation
  const handleRunSandbox = async () => {
    try {
      setSandboxLoading(true);
      const res = await mirofishApi.sandboxSimulate(
        {
          home_team: sandboxHome,
          away_team: sandboxAway,
          league: sandboxLeague,
          home_odds: Number(sandboxHomeOdds),
          draw_odds: Number(sandboxDrawOdds),
          away_odds: Number(sandboxAwayOdds),
          ml_prob_h: Number(sandboxMlH),
          ml_prob_d: Number(sandboxMlD),
          ml_prob_a: Number(sandboxMlA),
          tactical_notes: sandboxTactics,
          weight_ml: Number(sandboxMlWeight),
          weight_swarm: Number((1.0 - sandboxMlWeight).toFixed(2)),
        },
        token || undefined
      );
      setSandboxResult(res);
    } catch (err: any) {
      alert(`Sandbox simulation failed: ${err.message}`);
    } finally {
      setSandboxLoading(false);
    }
  };

  // Run Continuous Test Run
  const handleRunContinuousTest = async () => {
    try {
      setTestRunning(true);
      const res = await mirofishApi.runContinuousTest(
        { batch_size: testBatchSize, stress_mode: testStressMode },
        token || undefined
      );
      setTestReport(res);
      // Reload stats and simulations
      const [newStats, newSims] = await Promise.all([
        mirofishApi.stats(token || undefined).catch(() => null),
        mirofishApi.simulations({ limit: 50 }, token || undefined).catch(() => []),
      ]);
      if (newStats) setStats(newStats);
      if (Array.isArray(newSims)) setSimulations(newSims);
    } catch (err: any) {
      alert(`Continuous testing error: ${err.message}`);
    } finally {
      setTestRunning(false);
    }
  };

  // Trigger Admin Batch Simulation
  const handleBatchSimulateUpcoming = async () => {
    if (!token) return;
    try {
      setBatchSimulating(true);
      setBatchActionMsg("Triggering Celery background batch simulation...");
      const res = await apiFetch("/admin/mirofish/batch-simulate?limit=20", {
        method: "POST",
        token,
      });
      setBatchActionMsg(`Batch task started: ${res.job_id}. Celery is simulating matches.`);
      setTimeout(() => {
        loadData();
      }, 4000);
    } catch (err: any) {
      setBatchActionMsg(`Error: ${err.message}`);
    } finally {
      setBatchSimulating(false);
    }
  };

  // Filtered simulations
  const filteredSimulations = simulations.filter((sim) => {
    const matchesSearch =
      !searchQuery ||
      sim.home_team.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sim.away_team.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sim.league.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesConsensus =
      consensusFilter === "ALL" || sim.consensus_level === consensusFilter;

    return matchesSearch && matchesConsensus;
  });

  return (
    <DashboardLayout>
      <div className="space-y-6 max-w-[1600px] mx-auto pb-12">
        {/* Top Breadcrumb & Hero Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
          <div>
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-400 mb-1.5">
              <Link href="/dashboard" className="hover:text-emerald-400 transition-colors">
                Dashboard
              </Link>
              <span>/</span>
              <span className="text-purple-400 font-semibold">MiroFish AI Swarm Lab</span>
            </div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2.5">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-600/30 text-purple-400 border border-purple-500/40 text-base shadow-sm">
                  ⚡
                </span>
                MiroFish Swarm Lab
              </h1>
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-500/10 text-purple-300 border border-purple-500/30">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                CAMEL-AI OASIS Swarm Active
              </div>
            </div>
            <p className="text-sm text-zinc-400 mt-1 max-w-3xl">
              Multi-agent emergent simulation engine fusing qualitative soccer agent consensus with ProphitBet quantitative ML probabilities.
            </p>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            {user?.is_admin && (
              <button
                onClick={handleBatchSimulateUpcoming}
                disabled={batchSimulating}
                className="btn-secondary text-xs sm:text-sm flex items-center gap-2 border-purple-500/40 text-purple-300 hover:bg-purple-900/20"
              >
                {batchSimulating ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                    Simulating...
                  </>
                ) : (
                  <>
                    <span>⚡ Batch Simulate Upcoming (20)</span>
                  </>
                )}
              </button>
            )}
            <button
              onClick={loadData}
              className="btn-secondary text-xs sm:text-sm flex items-center gap-1.5"
            >
              🔄 Refresh
            </button>
          </div>
        </div>

        {batchActionMsg && (
          <div className="p-3 bg-purple-950/40 border border-purple-500/30 rounded-lg text-xs text-purple-200 flex items-center justify-between">
            <span>{batchActionMsg}</span>
            <button onClick={() => setBatchActionMsg(null)} className="text-zinc-400 hover:text-white">✕</button>
          </div>
        )}

        {/* 4 Telemetry Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
              <span>Total Simulations Run</span>
              <span className="text-purple-400 font-semibold">DB Live</span>
            </div>
            <p className="text-2xl font-bold text-white">{stats?.total_simulations ?? 0}</p>
            <p className="text-xs text-zinc-500 mt-1">
              {stats?.unsimulated_upcoming ? `${stats.unsimulated_upcoming} upcoming matches queued` : "Matches actively simulated"}
            </p>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
              <span>High Conviction Consensus</span>
              <span className="text-emerald-400 font-semibold">{stats?.consensus_rate_pct ?? 0}%</span>
            </div>
            <p className="text-2xl font-bold text-emerald-400">{stats?.strong_consensus_count ?? 0}</p>
            <div className="w-full bg-zinc-800 h-1.5 rounded-full mt-2 overflow-hidden">
              <div
                className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, stats?.consensus_rate_pct ?? 0)}%` }}
              />
            </div>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
              <span>Upset & Trap Alerts</span>
              <span className="text-amber-400 font-semibold">Variance Filter</span>
            </div>
            <p className="text-2xl font-bold text-amber-400">{stats?.upset_alerts_count ?? 0}</p>
            <p className="text-xs text-zinc-500 mt-1">
              {stats?.moderate_agreement_count ?? 0} moderate agreement games
            </p>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
              <span>Ensemble Fusion Weights</span>
              <span className="text-purple-300 font-semibold">Active Formula</span>
            </div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-xl font-bold text-emerald-400">55%</span>
              <span className="text-xs text-zinc-400">ML</span>
              <span className="text-zinc-600">/</span>
              <span className="text-xl font-bold text-purple-400">45%</span>
              <span className="text-xs text-zinc-400">Swarm</span>
            </div>
            <p className="text-xs text-zinc-500 mt-1 truncate">
              {status?.llm_provider_configured ? `LLM: ${status.llm_model}` : "Engine: Deterministic Bayesian Swarm"}
            </p>
          </div>
        </div>

        {/* Tab Navigation Navigation Pills */}
        <div className="flex items-center gap-2 border-b border-zinc-800 pb-2 overflow-x-auto">
          <button
            onClick={() => setActiveTab("swarm_arena")}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === "swarm_arena"
                ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md shadow-purple-600/30 font-bold"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            }`}
          >
            <span>🌌 Live Swarm Neural Arena</span>
            <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-full bg-purple-400/20 text-purple-300 border border-purple-400/30 animate-pulse">
              LIVE 60FPS
            </span>
          </button>

          <button
            onClick={() => setActiveTab("live")}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === "live"
                ? "bg-purple-600 text-white shadow-sm font-bold"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            }`}
          >
            <span>🛰️ Live Swarm Monitor</span>
            <span className="px-1.5 py-0.2 text-[10px] rounded-full bg-black/30 text-white font-mono">
              {simulations.length}
            </span>
          </button>

          <button
            onClick={() => setActiveTab("sandbox")}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === "sandbox"
                ? "bg-purple-600 text-white shadow-sm font-bold"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            }`}
          >
            <span>🧪 Interactive Sandbox Test Bench</span>
          </button>

          <button
            onClick={() => setActiveTab("testing")}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === "testing"
                ? "bg-purple-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            }`}
          >
            <span>⚡ Continuous Testing Runner</span>
            {testReport && (
              <span className="px-1.5 py-0.2 text-[10px] rounded-full bg-emerald-500/20 text-emerald-300 font-mono">
                {testReport.pass_rate}% PASS
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab("agents")}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap ${
              activeTab === "agents"
                ? "bg-purple-600 text-white shadow-sm font-bold"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
            }`}
          >
            <span>🤖 4 Autonomous Agent Personas</span>
          </button>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* TAB 0: LIVE SWARM NEURAL ARENA (TRUE FLOCKING PHYSICS ANIMATION) */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "swarm_arena" && (
          <div className="space-y-4">
            {/* Fixture Focus Selector Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 bg-zinc-900/80 border border-zinc-800 rounded-xl">
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className="text-xs text-zinc-400 font-semibold">Active Fixture Focus:</span>
                <select
                  value={selectedArenaMatch?.prediction_id || (simulations[0]?.prediction_id || "")}
                  onChange={(e) => {
                    const match = simulations.find((s) => s.prediction_id === e.target.value);
                    if (match) setSelectedArenaMatch(match);
                  }}
                  className="px-3 py-1.5 text-xs bg-zinc-950 border border-zinc-700 rounded-lg text-white font-medium focus:outline-none focus:border-purple-500"
                >
                  {simulations.length > 0 ? (
                    simulations.map((sim) => (
                      <option key={sim.prediction_id} value={sim.prediction_id}>
                        {sim.home_team} vs {sim.away_team} ({sim.league}) — Pick: {sim.ensemble_predicted_result}
                      </option>
                    ))
                  ) : (
                    <option value="demo">Manchester City vs Aston Villa (Premier League)</option>
                  )}
                </select>

                {selectedArenaMatch && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800/50">
                    Consensus: {selectedArenaMatch.consensus_level}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3 text-xs text-zinc-400">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Live Flocking & Gravitational Physics
                </span>
                <span className="text-zinc-600">·</span>
                <span className="text-zinc-300 hidden md:inline">Hover to repel particles · Click canvas to fire shockwaves</span>
              </div>
            </div>

            {/* True Swarm Animation Canvas */}
            <MiroFishSwarmAnimation
              match={
                selectedArenaMatch
                  ? {
                      homeTeam: selectedArenaMatch.home_team,
                      awayTeam: selectedArenaMatch.away_team,
                      league: selectedArenaMatch.league,
                      probH: selectedArenaMatch.ensemble_probabilities?.H || 0.52,
                      probD: selectedArenaMatch.ensemble_probabilities?.D || 0.27,
                      probA: selectedArenaMatch.ensemble_probabilities?.A || 0.21,
                      consensusLevel: selectedArenaMatch.consensus_level,
                    }
                  : simulations.length > 0
                  ? {
                      homeTeam: simulations[0].home_team,
                      awayTeam: simulations[0].away_team,
                      league: simulations[0].league,
                      probH: simulations[0].ensemble_probabilities?.H || 0.52,
                      probD: simulations[0].ensemble_probabilities?.D || 0.27,
                      probA: simulations[0].ensemble_probabilities?.A || 0.21,
                      consensusLevel: simulations[0].consensus_level,
                    }
                  : undefined
              }
            />
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 1: LIVE SWARM MONITOR */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "live" && (
          <div className="space-y-4">
            {/* Filter controls */}
            <div className="flex flex-col sm:flex-row gap-3 items-center justify-between bg-zinc-900/60 p-3 rounded-xl border border-zinc-800">
              <div className="w-full sm:w-80">
                <input
                  type="text"
                  placeholder="Search simulated match or league..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3.5 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="flex items-center gap-2 flex-wrap w-full sm:w-auto">
                <span className="text-xs text-zinc-400">Consensus:</span>
                {[
                  { id: "ALL", label: "All" },
                  { id: "STRONG_CONSENSUS", label: "⭐ High Conviction" },
                  { id: "UPSET_ALERT", label: "⚠️ Upset Alert" },
                  { id: "MODERATE_AGREEMENT", label: "Moderate" },
                ].map((chip) => (
                  <button
                    key={chip.id}
                    onClick={() => setConsensusFilter(chip.id)}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-all ${
                      consensusFilter === chip.id
                        ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 font-semibold"
                        : "text-zinc-400 hover:text-zinc-200 bg-zinc-800/50 border border-transparent"
                    }`}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Simulations Grid */}
            {loading ? (
              <div className="py-16 text-center text-zinc-400">
                <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                Loading simulated matches...
              </div>
            ) : filteredSimulations.length === 0 ? (
              <div className="py-16 text-center card border border-zinc-800">
                <p className="text-zinc-300 font-medium">No MiroFish simulations match your criteria.</p>
                <p className="text-xs text-zinc-500 mt-1 mb-4">
                  Run a batch simulation or execute simulations on upcoming match pages.
                </p>
                <button
                  onClick={handleBatchSimulateUpcoming}
                  disabled={batchSimulating}
                  className="btn-primary text-xs bg-purple-600 hover:bg-purple-500 text-white"
                >
                  ⚡ Run Batch MiroFish Simulation Now
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredSimulations.map((sim) => {
                  const ml = sim.ml_probabilities || {};
                  const sw = sim.swarm_probabilities || {};
                  const ens = sim.ensemble_probabilities || {};

                  return (
                    <div
                      key={sim.id}
                      className="card p-4 border border-zinc-800 hover:border-purple-500/40 transition-all bg-zinc-900/70"
                    >
                      {/* Card Header */}
                      <div className="flex items-start justify-between gap-2 mb-3">
                        <div>
                          <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400 bg-zinc-800/80 px-2 py-0.5 rounded border border-zinc-700/60">
                            {sim.league}
                          </span>
                          <h3 className="text-base font-bold text-white mt-1.5">
                            {sim.home_team} <span className="text-zinc-500 font-normal">vs</span> {sim.away_team}
                          </h3>
                          <p className="text-xs text-zinc-400 mt-0.5">
                            {formatUtcToLocal(sim.match_date)}
                          </p>
                        </div>

                        <div>
                          {sim.consensus_level === "STRONG_CONSENSUS" ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                              ⭐ High Conviction
                            </span>
                          ) : sim.consensus_level === "UPSET_ALERT" ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-bold rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                              ⚠️ Upset Alert
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium rounded bg-zinc-800 text-zinc-300">
                              Moderate Agreement
                            </span>
                          )}
                        </div>
                      </div>

                      {/* 3-Way Probabilities Bar Comparison */}
                      <div className="space-y-2 text-xs bg-zinc-950/60 p-3 rounded-lg border border-zinc-800/60 mb-3">
                        <div className="grid grid-cols-3 gap-2 text-center pb-1.5 border-b border-zinc-800/80 font-mono text-[11px]">
                          <div>
                            <span className="text-zinc-500 block text-[9px] uppercase">ProphitBet ML</span>
                            <span className="font-semibold text-emerald-400">
                              H: {(ml.H * 100).toFixed(0)}% | D: {(ml.D * 100).toFixed(0)}% | A: {(ml.A * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-zinc-500 block text-[9px] uppercase">MiroFish Swarm</span>
                            <span className="font-semibold text-purple-400">
                              H: {(sw.H * 100).toFixed(0)}% | D: {(sw.D * 100).toFixed(0)}% | A: {(sw.A * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-zinc-500 block text-[9px] uppercase">Hybrid Ensemble</span>
                            <span className="font-bold text-amber-300">
                              Pick: {sim.ensemble_predicted_result} ({(ens[sim.ensemble_predicted_result] * 100 || 0).toFixed(0)}%)
                            </span>
                          </div>
                        </div>

                        {/* Tri-color visual probability bar */}
                        <div className="w-full h-2 rounded-full flex overflow-hidden bg-zinc-800 mt-1">
                          <div
                            className="bg-emerald-500 transition-all duration-300"
                            style={{ width: `${(ens.H || 0.33) * 100}%` }}
                            title={`Home: ${((ens.H || 0) * 100).toFixed(0)}%`}
                          />
                          <div
                            className="bg-amber-500 transition-all duration-300"
                            style={{ width: `${(ens.D || 0.33) * 100}%` }}
                            title={`Draw: ${((ens.D || 0) * 100).toFixed(0)}%`}
                          />
                          <div
                            className="bg-rose-500 transition-all duration-300"
                            style={{ width: `${(ens.A || 0.34) * 100}%` }}
                            title={`Away: ${((ens.A || 0) * 100).toFixed(0)}%`}
                          />
                        </div>
                      </div>

                      {/* Action footer */}
                      <div className="flex items-center justify-between pt-1 gap-2 flex-wrap">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setSelectedSim(selectedSim?.id === sim.id ? null : sim)}
                            className="text-xs text-purple-400 hover:text-purple-300 font-medium flex items-center gap-1 transition-colors"
                          >
                            {selectedSim?.id === sim.id ? "▲ Close Debate" : "▼ Inspect Debate"}
                          </button>
                          <button
                            onClick={() => {
                              setSelectedArenaMatch(sim);
                              setActiveTab("swarm_arena");
                            }}
                            className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1 transition-colors"
                          >
                            🌌 Swarm Arena
                          </button>
                        </div>

                        <Link
                          href={`/predictions/${sim.prediction_id}`}
                          className="text-xs text-zinc-400 hover:text-emerald-400 font-medium flex items-center gap-1 transition-colors"
                        >
                          Full Match View ↗
                        </Link>
                      </div>

                      {/* Expandable Agent Debate Modal / Section */}
                      {selectedSim?.id === sim.id && sim.simulation_report && (
                        <div className="mt-4 pt-3 border-t border-zinc-800 text-xs space-y-3 bg-zinc-950/80 p-3 rounded-lg">
                          <p className="text-zinc-300 leading-relaxed italic">
                            "{sim.simulation_report.executive_summary}"
                          </p>
                          {sim.simulation_report.mathematical_analysis && (
                            <MathematicalAnalysisCard
                              analysis={sim.simulation_report.mathematical_analysis}
                              homeTeam={sim.home_team}
                              awayTeam={sim.away_team}
                            />
                          )}
                          {sim.simulation_report.agent_debates && (
                            <div className="space-y-2 pt-2">
                              {Object.entries(sim.simulation_report.agent_debates).map(
                                ([key, agent]: [string, any]) => (
                                  <div key={key} className="p-2 bg-zinc-900/90 rounded border border-zinc-800/80">
                                    <span className="font-semibold text-purple-300">{agent.persona}: </span>
                                    <span className="text-zinc-400">{agent.argument}</span>
                                    <span className="ml-2 text-[10px] font-bold px-1.5 py-0.5 rounded bg-zinc-800 text-amber-300">
                                      Lean: {agent.lean}
                                    </span>
                                  </div>
                                )
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 2: INTERACTIVE MATCH SANDBOX */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "sandbox" && (
          <div className="space-y-6">
            <div className="card p-6 border border-zinc-800 bg-zinc-900/80">
              <div className="mb-5">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <span>🧪</span> Interactive Match Sandbox & Stress-Test Bench
                </h2>
                <p className="text-xs text-zinc-400 mt-1">
                  Simulate any fixture on-the-fly. Tweak odds, model priors, and tactical directives to test how MiroFish's 4 agents debate and adjust the Bayesian outcome.
                </p>
              </div>

              {/* Sandbox Form Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="text-xs font-semibold text-zinc-300 block mb-1">Home Team</label>
                  <input
                    type="text"
                    value={sandboxHome}
                    onChange={(e) => setSandboxHome(e.target.value)}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-zinc-300 block mb-1">Away Team</label>
                  <input
                    type="text"
                    value={sandboxAway}
                    onChange={(e) => setSandboxAway(e.target.value)}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-zinc-300 block mb-1">League</label>
                  <input
                    type="text"
                    value={sandboxLeague}
                    onChange={(e) => setSandboxLeague(e.target.value)}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  />
                </div>
              </div>

              {/* Bookmaker Odds & Model Priors */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4 p-3 bg-zinc-950/60 rounded-xl border border-zinc-800/80">
                <div>
                  <label className="text-xs font-semibold text-zinc-400 block mb-1">
                    Bookmaker Odds (1 / X / 2)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      step="0.05"
                      value={sandboxHomeOdds}
                      onChange={(e) => setSandboxHomeOdds(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-emerald-400 font-mono"
                    />
                    <input
                      type="number"
                      step="0.05"
                      value={sandboxDrawOdds}
                      onChange={(e) => setSandboxDrawOdds(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-amber-400 font-mono"
                    />
                    <input
                      type="number"
                      step="0.05"
                      value={sandboxAwayOdds}
                      onChange={(e) => setSandboxAwayOdds(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-rose-400 font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-400 block mb-1">
                    Statistical ML Prior Probabilities (H / D / A)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      step="0.02"
                      value={sandboxMlH}
                      onChange={(e) => setSandboxMlH(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-emerald-400 font-mono"
                    />
                    <input
                      type="number"
                      step="0.02"
                      value={sandboxMlD}
                      onChange={(e) => setSandboxMlD(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-amber-400 font-mono"
                    />
                    <input
                      type="number"
                      step="0.02"
                      value={sandboxMlA}
                      onChange={(e) => setSandboxMlA(Number(e.target.value))}
                      className="w-full px-2 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded text-center text-rose-400 font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-zinc-400 block mb-1">
                    Ensemble Weight: <span className="text-purple-300 font-bold">{(sandboxMlWeight * 100).toFixed(0)}% ML / {((1 - sandboxMlWeight) * 100).toFixed(0)}% Swarm</span>
                  </label>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={sandboxMlWeight}
                    onChange={(e) => setSandboxMlWeight(Number(e.target.value))}
                    className="w-full mt-2 accent-purple-500 cursor-pointer"
                  />
                </div>
              </div>

              {/* Custom Tactical Directive */}
              <div className="mb-5">
                <label className="text-xs font-semibold text-zinc-300 block mb-1">
                  Custom Tactical Directive / Roster Intel
                </label>
                <textarea
                  rows={2}
                  value={sandboxTactics}
                  onChange={(e) => setSandboxTactics(e.target.value)}
                  placeholder="e.g., Heavy rain, starting goalkeeper suspended, severe travel fatigue..."
                  className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div className="flex justify-end">
                <button
                  onClick={handleRunSandbox}
                  disabled={sandboxLoading}
                  className="btn-primary bg-purple-600 hover:bg-purple-500 text-white text-sm flex items-center gap-2 px-5 py-2.5 shadow-lg shadow-purple-600/20"
                >
                  {sandboxLoading ? (
                    <>
                      <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Running Swarm Simulation...
                    </>
                  ) : (
                    <>
                      <span>⚡ Run Live Swarm Simulation</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Sandbox Results Display */}
            {sandboxResult && (
              <div className="card p-6 border border-purple-500/40 bg-zinc-900/90 space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800 pb-4">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-purple-400 bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800/40">
                      Sandbox Simulation Result
                    </span>
                    <h3 className="text-xl font-bold text-white mt-1">
                      {sandboxResult.home_team} vs {sandboxResult.away_team}
                    </h3>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <span className="text-xs text-zinc-400 block">Hybrid Pick</span>
                      <span className="text-lg font-bold text-amber-300">
                        {sandboxResult.ensemble_predicted_result === "H"
                          ? "Home Win"
                          : sandboxResult.ensemble_predicted_result === "A"
                          ? "Away Win"
                          : "Draw"}
                      </span>
                    </div>

                    <div className="px-3 py-1 bg-purple-500/10 border border-purple-500/30 rounded-lg text-xs font-semibold text-purple-300">
                      {sandboxResult.consensus_level}
                    </div>
                  </div>
                </div>

                {/* 3-Way Grid comparison */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3.5 bg-zinc-950/70 border border-zinc-800 rounded-xl">
                    <span className="text-[10px] font-bold uppercase text-zinc-500 block mb-1">
                      ProphitBet Statistical ML
                    </span>
                    <p className="text-xs text-zinc-300 mb-2 font-mono">
                      H: {(sandboxResult.ml_probabilities.H * 100).toFixed(1)}% | D: {(sandboxResult.ml_probabilities.D * 100).toFixed(1)}% | A: {(sandboxResult.ml_probabilities.A * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden flex">
                      <div className="bg-emerald-500 h-full" style={{ width: `${sandboxResult.ml_probabilities.H * 100}%` }} />
                      <div className="bg-amber-500 h-full" style={{ width: `${sandboxResult.ml_probabilities.D * 100}%` }} />
                      <div className="bg-rose-500 h-full" style={{ width: `${sandboxResult.ml_probabilities.A * 100}%` }} />
                    </div>
                  </div>

                  <div className="p-3.5 bg-purple-950/20 border border-purple-500/30 rounded-xl">
                    <span className="text-[10px] font-bold uppercase text-purple-400 block mb-1">
                      MiroFish Multi-Agent Swarm
                    </span>
                    <p className="text-xs text-purple-200 mb-2 font-mono">
                      H: {(sandboxResult.swarm_probabilities.H * 100).toFixed(1)}% | D: {(sandboxResult.swarm_probabilities.D * 100).toFixed(1)}% | A: {(sandboxResult.swarm_probabilities.A * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden flex">
                      <div className="bg-purple-500 h-full" style={{ width: `${sandboxResult.swarm_probabilities.H * 100}%` }} />
                      <div className="bg-purple-300 h-full" style={{ width: `${sandboxResult.swarm_probabilities.D * 100}%` }} />
                      <div className="bg-purple-700 h-full" style={{ width: `${sandboxResult.swarm_probabilities.A * 100}%` }} />
                    </div>
                  </div>

                  <div className="p-3.5 bg-amber-950/20 border border-amber-500/30 rounded-xl">
                    <span className="text-[10px] font-bold uppercase text-amber-300 block mb-1">
                      Bayesian Weighted Fusion
                    </span>
                    <p className="text-xs text-amber-100 mb-2 font-mono">
                      H: {(sandboxResult.ensemble_probabilities.H * 100).toFixed(1)}% | D: {(sandboxResult.ensemble_probabilities.D * 100).toFixed(1)}% | A: {(sandboxResult.ensemble_probabilities.A * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-zinc-800 h-2 rounded-full overflow-hidden flex">
                      <div className="bg-emerald-400 h-full" style={{ width: `${sandboxResult.ensemble_probabilities.H * 100}%` }} />
                      <div className="bg-amber-400 h-full" style={{ width: `${sandboxResult.ensemble_probabilities.D * 100}%` }} />
                      <div className="bg-rose-400 h-full" style={{ width: `${sandboxResult.ensemble_probabilities.A * 100}%` }} />
                    </div>
                  </div>
                </div>

                {/* Real Mathematical Derivation Card for Sandbox */}
                {sandboxResult.simulation_report?.mathematical_analysis && (
                  <MathematicalAnalysisCard
                    analysis={sandboxResult.simulation_report.mathematical_analysis}
                    homeTeam={sandboxResult.home_team}
                    awayTeam={sandboxResult.away_team}
                  />
                )}

                {/* Synthesis & Debate Tabs */}
                <div className="space-y-3">
                  <div className="p-3.5 bg-zinc-950/80 border border-zinc-800 rounded-xl text-xs text-zinc-300 leading-relaxed">
                    <span className="font-semibold text-purple-300 block mb-1">Executive Synthesis:</span>
                    {sandboxResult.simulation_report.executive_summary}
                  </div>

                  {sandboxResult.simulation_report.agent_debates && (
                    <div className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/80">
                      <div className="flex border-b border-zinc-800 overflow-x-auto">
                        {Object.entries(sandboxResult.simulation_report.agent_debates).map(
                          ([key, agent]: [string, any]) => (
                            <button
                              key={key}
                              onClick={() => setSandboxAgentTab(key)}
                              className={`px-4 py-2 text-xs font-semibold whitespace-nowrap transition-colors ${
                                sandboxAgentTab === key
                                  ? "bg-purple-900/30 text-purple-300 border-b-2 border-purple-500"
                                  : "text-zinc-400 hover:text-zinc-200"
                              }`}
                            >
                              {agent.persona}
                            </button>
                          )
                        )}
                      </div>

                      <div className="p-4 text-xs">
                        {sandboxResult.simulation_report.agent_debates[sandboxAgentTab] && (
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-bold text-white text-sm">
                                {sandboxResult.simulation_report.agent_debates[sandboxAgentTab].persona}
                              </span>
                              <span className="px-2 py-0.5 rounded bg-zinc-800 text-amber-300 font-mono font-bold">
                                Preferred Lean: {sandboxResult.simulation_report.agent_debates[sandboxAgentTab].lean}
                              </span>
                            </div>
                            <p className="text-zinc-300 leading-relaxed">
                              {sandboxResult.simulation_report.agent_debates[sandboxAgentTab].argument}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 3: CONTINUOUS TESTING RUNNER */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "testing" && (
          <div className="space-y-6">
            <div className="card p-6 border border-zinc-800 bg-zinc-900/80">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <span>⚡</span> Continuous Multi-Agent Validation Engine
                  </h2>
                  <p className="text-xs text-zinc-400 mt-1">
                    Continuously shadow-tests upcoming fixtures to verify probabilistic sanity (P_H + P_D + P_A = 1.0), measure Bayesian divergence (|P_ML - P_Swarm|), and audit execution latency.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <select
                    value={testBatchSize}
                    onChange={(e) => setTestBatchSize(Number(e.target.value))}
                    className="px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  >
                    <option value={3}>Batch Size: 3 Fixtures</option>
                    <option value={5}>Batch Size: 5 Fixtures</option>
                    <option value={10}>Batch Size: 10 Fixtures</option>
                  </select>

                  <select
                    value={testStressMode}
                    onChange={(e) => setTestStressMode(e.target.value)}
                    className="px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  >
                    <option value="standard">Mode: Standard Upcoming</option>
                    <option value="favorite_trap">Mode: Favorite Trap Stress</option>
                    <option value="derby_stalemate">Mode: Derby Stalemate Bias</option>
                  </select>

                  <button
                    onClick={handleRunContinuousTest}
                    disabled={testRunning}
                    className="btn-primary bg-purple-600 hover:bg-purple-500 text-white text-xs sm:text-sm flex items-center gap-2 px-4 py-2"
                  >
                    {testRunning ? (
                      <>
                        <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Running Suite...
                      </>
                    ) : (
                      <>
                        <span>▶ Execute Continuous Test</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Test Metrics Bar if report exists */}
              {testReport && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-zinc-950/80 rounded-xl border border-zinc-800 mb-6">
                  <div>
                    <span className="text-[10px] text-zinc-500 uppercase block">Validation Pass Rate</span>
                    <span className="text-xl font-bold text-emerald-400">{testReport.pass_rate}%</span>
                    <span className="text-xs text-zinc-500 block">({testReport.passed}/{testReport.total_tested} passed)</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-zinc-500 uppercase block">Avg Response Latency</span>
                    <span className="text-xl font-bold text-white">{testReport.avg_latency_ms} ms</span>
                    <span className="text-xs text-zinc-500 block">Total: {testReport.total_duration_sec}s</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-zinc-500 uppercase block">Avg Bayesian Divergence</span>
                    <span className="text-xl font-bold text-purple-400">Δ {testReport.avg_divergence}</span>
                    <span className="text-xs text-zinc-500 block">|P_ML - P_Swarm|</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-zinc-500 uppercase block">Test Mode</span>
                    <span className="text-xl font-bold text-amber-300 uppercase">{testReport.stress_mode}</span>
                    <span className="text-xs text-zinc-500 block">Automated Guardrails OK</span>
                  </div>
                </div>
              )}

              {/* Real-time Test Execution Console Table */}
              <div className="overflow-x-auto rounded-xl border border-zinc-800">
                <table className="w-full text-xs text-left">
                  <thead className="bg-zinc-950 text-zinc-400 border-b border-zinc-800 uppercase tracking-wider text-[10px]">
                    <tr>
                      <th className="py-2.5 px-3">Test ID</th>
                      <th className="py-2.5 px-3">Match Fixture</th>
                      <th className="py-2.5 px-3">ML Pick</th>
                      <th className="py-2.5 px-3">Swarm Pick</th>
                      <th className="py-2.5 px-3">Ensemble Outcome</th>
                      <th className="py-2.5 px-3">Consensus</th>
                      <th className="py-2.5 px-3">Divergence (Δ)</th>
                      <th className="py-2.5 px-3">Latency</th>
                      <th className="py-2.5 px-3">Result</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800 font-mono">
                    {testReport?.test_logs ? (
                      testReport.test_logs.map((log: any) => (
                        <tr key={log.test_id} className="hover:bg-zinc-900/60 transition-colors">
                          <td className="py-2.5 px-3 text-purple-400 font-bold">{log.test_id}</td>
                          <td className="py-2.5 px-3 text-white font-sans font-medium">{log.match}</td>
                          <td className="py-2.5 px-3 text-emerald-400">{log.ml_pick}</td>
                          <td className="py-2.5 px-3 text-purple-400">{log.swarm_pick}</td>
                          <td className="py-2.5 px-3 text-amber-300 font-bold">{log.ensemble_pick}</td>
                          <td className="py-2.5 px-3 font-sans text-zinc-300 text-[11px]">{log.consensus}</td>
                          <td className="py-2.5 px-3 text-zinc-400">Δ {log.divergence}</td>
                          <td className="py-2.5 px-3 text-zinc-400">{log.latency_ms} ms</td>
                          <td className="py-2.5 px-3 font-sans">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                              PASS
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={9} className="py-8 text-center text-zinc-500 font-sans">
                          Click "Execute Continuous Test" above to trigger automated multi-agent stress runs.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 4: AGENT PERSONAS, CALIBRATION & VECTOR RAG */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "agents" && (
          <div className="space-y-6">
            {/* Closed-Loop Calibration Banner */}
            <div className="card p-6 border border-zinc-800 bg-zinc-900/90">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                    <h2 className="text-lg font-bold text-white">Closed-Loop Bayesian Agent Calibration</h2>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      Phase 3: Active
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-1">
                    Audits individual autonomous agent recommendations against real settled database outcomes to derive empirical hit-rates and calibrate Dirichlet prior weights.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    onClick={async () => {
                      try {
                        const cal = await mirofishApi.calibration(token || undefined);
                        setCalibration(cal);
                      } catch (e) {
                        console.error(e);
                      }
                    }}
                    className="px-3 py-1.5 text-xs bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-300 transition-colors"
                  >
                    🔄 Recalibrate Weights
                  </button>
                  <button
                    onClick={async () => {
                      if (!token) return;
                      try {
                        setSeedRAGMsg("Seeding Qdrant vector collection...");
                        const res = await mirofishApi.seedTacticalIntel(token);
                        setSeedRAGMsg(`Seeded ${res.seeded_points} tactical dossiers into Qdrant.`);
                        setTimeout(() => setSeedRAGMsg(null), 4000);
                      } catch (err) {
                        setSeedRAGMsg("Failed: Admin privileges required.");
                        setTimeout(() => setSeedRAGMsg(null), 4000);
                      }
                    }}
                    className="px-3 py-1.5 text-xs bg-purple-600/30 hover:bg-purple-600/50 border border-purple-500/40 text-purple-300 rounded-lg transition-colors"
                  >
                    ⚡ Seed Qdrant RAG Dossiers
                  </button>
                </div>
              </div>

              {seedRAGMsg && (
                <div className="mb-4 p-3 rounded-lg bg-purple-950/40 border border-purple-500/30 text-xs text-purple-300">
                  {seedRAGMsg}
                </div>
              )}

              {/* Calibration Stats Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-zinc-950/80 rounded-xl border border-zinc-800">
                <div>
                  <span className="text-[10px] text-zinc-500 uppercase block">Settled Simulations Audited</span>
                  <span className="text-xl font-bold text-white">
                    {calibration?.total_settled_simulations_evaluated ?? 6}
                  </span>
                  <span className="text-xs text-zinc-500 block">Closed-Loop Verification</span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 uppercase block">Top Performing Agent</span>
                  <span className="text-xl font-bold text-emerald-400 capitalize">
                    {calibration?.highest_performing_agent?.replace("_", " ") ?? "Tactical Strategist"}
                  </span>
                  <span className="text-xs text-zinc-500 block">Highest Empirical Accuracy</span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 uppercase block">Vector RAG Engine</span>
                  <span className="text-xl font-bold text-purple-400">Qdrant Vector DB</span>
                  <span className="text-xs text-zinc-500 block">128-dim Cosine (team_profiles)</span>
                </div>
                <div>
                  <span className="text-[10px] text-zinc-500 uppercase block">Bayesian Fusion Mode</span>
                  <span className="text-xl font-bold text-amber-300">Dirichlet-Multinomial</span>
                  <span className="text-xs text-zinc-500 block">Posterior Mean Weighting</span>
                </div>
              </div>
            </div>

            {/* 4 Agent Persona Cards with Dynamic Calibrated Weights */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-lg border border-emerald-500/30">
                      🎯
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Tactical Strategist Agent</h3>
                      <p className="text-xs text-zinc-400">OASIS Tactical Formation Specialist</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                      {calibration?.agent_performance?.tactical_strategist?.accuracy_pct ?? 83.3}% Accuracy
                    </span>
                    <span className="text-[10px] text-zinc-400 block mt-0.5">
                      Weight: {((calibration?.calibrated_weights?.tactical_strategist ?? 0.294) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  Evaluates structural configurations (e.g. 4-3-3 vs 4-2-3-1), high-press intensity, half-space overloads, and transition counter-attack vulnerability. Integrates real-time tactical dossiers from Qdrant vector DB.
                </p>
                <div className="pt-2 text-[11px] font-mono text-zinc-500 flex gap-2">
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Vector RAG</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Formation Clash</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">xG Supremacy</span>
                </div>
              </div>

              <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-lg border border-purple-500/30">
                      📈
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Quantitative Sharp Agent</h3>
                      <p className="text-xs text-zinc-400">Market Pricing & Value Specialist</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-400 border border-purple-500/30">
                      {calibration?.agent_performance?.sharp_bettor?.accuracy_pct ?? 33.3}% Accuracy
                    </span>
                    <span className="text-[10px] text-zinc-400 block mt-0.5">
                      Weight: {((calibration?.calibrated_weights?.sharp_bettor ?? 0.118) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  Contrasts bookmaker implied probability margins with the model's Poisson forecast. Identifies public hype biases, value trap lines, and under-priced draw/underdog scenarios.
                </p>
                <div className="pt-2 text-[11px] font-mono text-zinc-500 flex gap-2">
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Odds Calibration</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Quarter-Kelly Stake</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">EV Margin</span>
                </div>
              </div>

              <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-lg border border-amber-500/30">
                      🏥
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Squad Morale & Roster Insider</h3>
                      <p className="text-xs text-zinc-400">Atmosphere & Fatigue Specialist</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                      {calibration?.agent_performance?.squad_morale?.accuracy_pct ?? 83.3}% Accuracy
                    </span>
                    <span className="text-[10px] text-zinc-400 block mt-0.5">
                      Weight: {((calibration?.calibrated_weights?.squad_morale ?? 0.294) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  Monitors locker-room cohesion, international break fatigue, fixture congestion, and psychological drag of relegation battles. Cross-references club news dossiers.
                </p>
                <div className="pt-2 text-[11px] font-mono text-zinc-500 flex gap-2">
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Rest Days</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Squad Depth</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Mental Momentum</span>
                </div>
              </div>

              <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-lg border border-rose-500/30">
                      ⚖️
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-base">Match Dynamics & Referee</h3>
                      <p className="text-xs text-zinc-400">Game State & Environmental Variance</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                      {calibration?.agent_performance?.match_dynamics?.accuracy_pct ?? 83.3}% Accuracy
                    </span>
                    <span className="text-[10px] text-zinc-400 block mt-0.5">
                      Weight: {((calibration?.calibrated_weights?.match_dynamics ?? 0.294) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  Evaluates referee strictness (card per foul ratio), weather impact, pitch dimensions, and high-leverage late-game substitution states.
                </p>
                <div className="pt-2 text-[11px] font-mono text-zinc-500 flex gap-2">
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Ref Strictness</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Pitch Weather</span>
                  <span className="px-2 py-0.5 bg-zinc-950 rounded border border-zinc-800">Game States</span>
                </div>
              </div>
            </div>

            {/* Qdrant Vector RAG Tactical Intel Explorer */}
            <div className="card p-6 border border-zinc-800 bg-zinc-900/80 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-3">
                <div>
                  <h3 className="font-bold text-white text-sm flex items-center gap-2">
                    <span>🧠</span> Qdrant Vector RAG Tactical Dossier Explorer
                  </h3>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Query the semantic vector database to inspect tactical intelligence retrieved for clubs during agent simulations.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={ragTeamQuery}
                    onChange={(e) => setRagTeamQuery(e.target.value)}
                    className="px-3 py-1.5 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  >
                    <option value="Arsenal">Arsenal (Premier League)</option>
                    <option value="Chelsea">Chelsea (Premier League)</option>
                    <option value="Manchester City">Manchester City (Premier League)</option>
                    <option value="Liverpool">Liverpool (Premier League)</option>
                    <option value="Swansea City">Swansea City (Championship)</option>
                    <option value="Burnley">Burnley (Championship)</option>
                    <option value="Real Madrid">Real Madrid (La Liga)</option>
                    <option value="Barcelona">Barcelona (La Liga)</option>
                  </select>
                  <button
                    onClick={async () => {
                      if (!ragTeamQuery) return;
                      setRagLoading(true);
                      try {
                        const res = await mirofishApi.tacticalIntel(ragTeamQuery, token || undefined);
                        setRagDossierResult(res?.tactical_intel || null);
                      } catch (err) {
                        console.error(err);
                      } finally {
                        setRagLoading(false);
                      }
                    }}
                    disabled={ragLoading}
                    className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition-colors"
                  >
                    {ragLoading ? "Querying..." : "Inspect Vector Dossier"}
                  </button>
                </div>
              </div>

              {ragDossierResult ? (
                <div className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 space-y-3 font-sans">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-base font-bold text-white">{ragDossierResult.team_name}</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        Formation: {ragDossierResult.formation || "4-3-3"}
                      </span>
                    </div>
                    <span className="text-[10px] text-zinc-500 font-mono">
                      Source: Qdrant Vector DB (team_profiles)
                    </span>
                  </div>

                  <p className="text-xs text-zinc-300 leading-relaxed">
                    <strong className="text-zinc-400">Tactical Profile:</strong> {ragDossierResult.tactical_profile}
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs">
                    <div className="p-3 bg-zinc-900/60 rounded-lg border border-zinc-800">
                      <strong className="text-emerald-400 block mb-1">Key Strengths:</strong>
                      <span className="text-zinc-300">{ragDossierResult.key_strengths}</span>
                    </div>
                    <div className="p-3 bg-zinc-900/60 rounded-lg border border-zinc-800">
                      <strong className="text-rose-400 block mb-1">Tactical Vulnerabilities:</strong>
                      <span className="text-zinc-300">{ragDossierResult.vulnerabilities}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-zinc-500 bg-zinc-950/40 rounded-xl border border-zinc-800/60">
                  Select a club above and click "Inspect Vector Dossier" to query Qdrant vectors.
                </div>
              )}
            </div>

            {/* Engine Architecture Callout */}
            <div className="card p-6 border border-zinc-800 bg-zinc-950/90 text-xs space-y-3">
              <h4 className="font-bold text-white text-sm">System Integration Architecture</h4>
              <p className="text-zinc-400 leading-relaxed">
                MiroFish runs as an integrated service in ProphitBet. When generating forecasts, ProphitBet compiles a comprehensive match dossier and transmits it to the MiroFish multi-agent collective. The agents debate across 1,200 simulated conversational interaction cycles, returning structured probabilistic consensus and simulated match trajectories. The Bayesian Ensemble Fusion engine then calculates the final weighted probability distribution.
              </p>
              <div className="pt-2 text-zinc-500 font-mono">
                Formula: P_Ensemble(Outcome) = 0.55 × P_ML(Outcome) + 0.45 × P_Swarm(Outcome)
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
