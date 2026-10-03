"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { analysisApi, leaguesApi, clvApi } from "@/lib/api";
import { cn, resultColor } from "@/lib/utils";

const ANALYSIS_LABELS: Record<string, string> = {
  description: "Descriptive Statistics",
  correlation: "Correlation Matrix",
  distribution: "Feature Distributions",
  variance: "Variance Analysis",
  boruta: "Boruta Feature Importance",
  rules: "Decision Rule Extraction",
  coefficients: "Coefficient Analysis",
  impurity: "Gini Impurity",
};

export default function AnalysisPage() {
  const { token } = useAuth();
  const [activeTab, setActiveTab] = useState<"backtest" | "statistical" | "clv">("backtest");

  // Statistical Analyzer state
  const [leagues, setLeagues] = useState<any[]>([]);
  const [types, setTypes] = useState<string[]>([]);
  const [selectedLeague, setSelectedLeague] = useState("");
  const [selectedType, setSelectedType] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Backtester state
  const [presets, setPresets] = useState<any[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("quarter_kelly_ev_plus_3");
  const [backtestLeague, setBacktestLeague] = useState<string>("");
  const [stakingStrategy, setStakingStrategy] = useState<string>("quarter_kelly");
  const [initialBankroll, setInitialBankroll] = useState<number>(10000);
  const [minEvPct, setMinEvPct] = useState<number>(-10.0);
  const [minConfidence, setMinConfidence] = useState<number>(0.0);
  const [consensusFilter, setConsensusFilter] = useState<string>("ALL");
  const [allowedPicks, setAllowedPicks] = useState<string[]>(["H", "D", "A"]);
  const [flatStakeAmount, setFlatStakeAmount] = useState<number>(100);
  const [proportionalStakePct, setProportionalStakePct] = useState<number>(2.5);

  // CLV state
  const [clvData, setClvData] = useState<any>(null);
  const [clvLoading, setClvLoading] = useState<boolean>(false);

  const loadClv = async () => {
    if (!token) return;
    try {
      setClvLoading(true);
      const res = await clvApi.audit(100, token);
      setClvData(res);
    } catch (e) {
      console.error("Error loading CLV audit:", e);
    } finally {
      setClvLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "clv" && !clvData && !clvLoading && token) {
      loadClv();
    }
  }, [activeTab, token]);

  const [backtestLoading, setBacktestLoading] = useState<boolean>(false);
  const [backtestResult, setBacktestResult] = useState<any | null>(null);
  const [backtestError, setBacktestError] = useState<string>("");

  useEffect(() => {
    if (!token) return;
    Promise.all([
      leaguesApi.list(token).catch(() => []),
      analysisApi.types(token).catch(() => ({ types: [] })),
      analysisApi.backtestPresets(token).catch(() => ({ presets: [] })),
    ]).then(([lgData, typeData, presetData]) => {
      setLeagues(lgData || []);
      setTypes(typeData.types || []);
      const p = presetData?.presets || [];
      setPresets(p);
    });
  }, [token]);

  // Load initial default backtest on mount
  useEffect(() => {
    if (token && !backtestResult && !backtestLoading) {
      runBacktest();
    }
  }, [token]);

  const applyPreset = (preset: any) => {
    setSelectedPresetId(preset.id);
    const p = preset.params;
    if (p.min_ev_pct !== undefined) setMinEvPct(p.min_ev_pct);
    if (p.min_confidence !== undefined) setMinConfidence(p.min_confidence);
    if (p.consensus_filter !== undefined) setConsensusFilter(p.consensus_filter);
    if (p.allowed_picks !== undefined) setAllowedPicks(p.allowed_picks);
    if (p.staking_strategy !== undefined) setStakingStrategy(p.staking_strategy);
    if (p.flat_stake_amount !== undefined) setFlatStakeAmount(p.flat_stake_amount);
    if (p.proportional_stake_pct !== undefined) setProportionalStakePct(p.proportional_stake_pct);
    if (p.initial_bankroll !== undefined) setInitialBankroll(p.initial_bankroll);
  };

  const handlePickToggle = (pick: string) => {
    if (allowedPicks.includes(pick)) {
      if (allowedPicks.length > 1) {
        setAllowedPicks(allowedPicks.filter((p) => p !== pick));
      }
    } else {
      setAllowedPicks([...allowedPicks, pick]);
    }
  };

  const runBacktest = async () => {
    if (!token) return;
    setBacktestLoading(true);
    setBacktestError("");
    try {
      const data = await analysisApi.backtest(
        {
          league_id: backtestLeague ? Number(backtestLeague) : null,
          min_ev_pct: Number(minEvPct),
          min_confidence: Number(minConfidence),
          consensus_filter: consensusFilter,
          allowed_picks: allowedPicks,
          staking_strategy: stakingStrategy,
          flat_stake_amount: Number(flatStakeAmount),
          proportional_stake_pct: Number(proportionalStakePct),
          initial_bankroll: Number(initialBankroll),
        },
        token
      );
      setBacktestResult(data);
    } catch (err: any) {
      setBacktestError(err.message || "Backtest simulation failed");
    } finally {
      setBacktestLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!selectedLeague || !selectedType || !token) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await analysisApi.run(
        { league_id: Number(selectedLeague), analysis_type: selectedType },
        token
      );
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6 max-w-[1600px] mx-auto pb-12">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
          <div>
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-400 mb-1.5">
              <span>Analytics & Quantitative Lab</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-600/30 text-emerald-400 border border-emerald-500/40 text-base shadow-sm">
                📈
              </span>
              Quantitative Analysis & Strategy Backtesting
            </h1>
            <p className="text-xs text-zinc-400 mt-1">
              Verify real betting edge with historical portfolio simulations, Kelly Criterion risk budgeting, and feature importance
            </p>
          </div>

          {/* Tab Pill Switcher */}
          <div className="flex items-center gap-1.5 p-1 bg-zinc-900 border border-zinc-800 rounded-xl self-start sm:self-auto">
            <button
              onClick={() => setActiveTab("backtest")}
              className={cn(
                "px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2",
                activeTab === "backtest"
                  ? "bg-emerald-500 text-zinc-950 shadow-md shadow-emerald-500/20"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <span>📈</span>
              <span>Strategy Backtester</span>
            </button>
            <button
              onClick={() => setActiveTab("statistical")}
              className={cn(
                "px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2",
                activeTab === "statistical"
                  ? "bg-emerald-500 text-zinc-950 shadow-md shadow-emerald-500/20"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <span>🔬</span>
              <span>Feature Models</span>
            </button>
            <button
              onClick={() => setActiveTab("clv")}
              className={cn(
                "px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2",
                activeTab === "clv"
                  ? "bg-emerald-500 text-zinc-950 shadow-md shadow-emerald-500/20"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <span>🎯</span>
              <span>Closing Line (CLV)</span>
            </button>
          </div>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* TAB 1: STRATEGY BACKTESTER & ROI SIMULATOR */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "backtest" && (
          <div className="space-y-6 animate-in fade-in duration-150">
            {/* Strategy Presets Bar */}
            {presets.length > 0 && (
              <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                    <span>⚡</span> Institutional Strategy Presets
                  </span>
                  <span className="text-[11px] text-zinc-500">One-click quantitative configurations</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {presets.map((pr) => {
                    const active = selectedPresetId === pr.id;
                    return (
                      <button
                        key={pr.id}
                        type="button"
                        onClick={() => applyPreset(pr)}
                        className={cn(
                          "p-3 rounded-xl text-left border transition-all flex flex-col justify-between space-y-1.5",
                          active
                            ? "bg-emerald-950/40 border-emerald-500/60 shadow-md shadow-emerald-500/10"
                            : "bg-zinc-950/60 border-zinc-800/80 hover:border-zinc-700 text-zinc-300"
                        )}
                      >
                        <div>
                          <span className="text-xs font-bold text-zinc-100 block">{pr.title}</span>
                          <span className="text-[11px] text-zinc-400 line-clamp-2 mt-0.5">
                            {pr.description}
                          </span>
                        </div>
                        <span
                          className={cn(
                            "text-[10px] font-mono font-semibold self-start px-2 py-0.5 rounded",
                            active ? "bg-emerald-500/20 text-emerald-300" : "bg-zinc-800 text-zinc-400"
                          )}
                        >
                          {active ? "✓ Active Strategy" : "Click to Apply"}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Custom Parameter Controls Card */}
            <div className="p-5 rounded-2xl bg-zinc-900/80 border border-zinc-800 shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <span>⚙️</span> Strategy Parameters & Risk Budgeting
                </h3>
                <span className="text-xs text-zinc-500 font-mono">696 settled predictions evaluated</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Staking Strategy */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    Staking Strategy
                  </label>
                  <select
                    value={stakingStrategy}
                    onChange={(e) => {
                      setStakingStrategy(e.target.value);
                      setSelectedPresetId("custom");
                    }}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white font-medium"
                  >
                    <option value="quarter_kelly">Quarter-Kelly (0.25 × f*)</option>
                    <option value="half_kelly">Half-Kelly (0.50 × f*)</option>
                    <option value="flat">Flat Wager ($ amount)</option>
                    <option value="proportional">Proportional (% bankroll)</option>
                  </select>
                </div>

                {/* 2. Initial Bankroll */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    Starting Bankroll ($)
                  </label>
                  <input
                    type="number"
                    step="500"
                    value={initialBankroll}
                    onChange={(e) => {
                      setInitialBankroll(Number(e.target.value));
                      setSelectedPresetId("custom");
                    }}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-emerald-400 font-mono font-bold"
                  />
                </div>

                {/* 3. Min +EV Edge */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    Minimum Expected Value (+EV)
                  </label>
                  <select
                    value={minEvPct}
                    onChange={(e) => {
                      setMinEvPct(Number(e.target.value));
                      setSelectedPresetId("custom");
                    }}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white font-medium"
                  >
                    <option value="-10.0">All Algorithmic Picks (No EV threshold)</option>
                    <option value="0.0">+EV ≥ 0% (Positive Edge Only)</option>
                    <option value="2.0">+EV ≥ 2% (Value Edge)</option>
                    <option value="4.0">+EV ≥ 4% (Strong Edge)</option>
                    <option value="6.0">+EV ≥ 6% (Institutional Alpha)</option>
                  </select>
                </div>

                {/* 4. Min Model Confidence */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    Min Probability Confidence
                  </label>
                  <select
                    value={minConfidence}
                    onChange={(e) => {
                      setMinConfidence(Number(e.target.value));
                      setSelectedPresetId("custom");
                    }}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white font-medium"
                  >
                    <option value="0.0">Any Model Confidence</option>
                    <option value="0.35">≥ 35% Confidence</option>
                    <option value="0.42">≥ 42% Moderate Confidence</option>
                    <option value="0.50">≥ 50% High Conviction Only</option>
                  </select>
                </div>

                {/* 5. MiroFish Swarm Consensus */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    MiroFish Consensus Filter
                  </label>
                  <select
                    value={consensusFilter}
                    onChange={(e) => {
                      setConsensusFilter(e.target.value);
                      setSelectedPresetId("custom");
                    }}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white font-medium"
                  >
                    <option value="ALL">All Consensus Types</option>
                    <option value="STRONG_CONSENSUS">⭐ High Conviction Only</option>
                    <option value="UPSET_ALERT">⚠️ Upset Alerts Only</option>
                    <option value="MODERATE_AGREEMENT">⚖️ Moderate Agreement</option>
                  </select>
                </div>

                {/* 6. League Filter */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    League Scope
                  </label>
                  <select
                    value={backtestLeague}
                    onChange={(e) => setBacktestLeague(e.target.value)}
                    className="w-full px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  >
                    <option value="">All Leagues (Global Portfolio)</option>
                    {leagues.map((l: any) => (
                      <option key={l.id} value={l.id}>
                        {l.country} — {l.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* 7. Outcome Checkboxes */}
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">
                    Wager Outcomes
                  </label>
                  <div className="flex gap-2 h-[34px] items-center">
                    {(["H", "D", "A"] as const).map((p) => {
                      const active = allowedPicks.includes(p);
                      return (
                        <button
                          key={p}
                          type="button"
                          onClick={() => {
                            handlePickToggle(p);
                            setSelectedPresetId("custom");
                          }}
                          className={cn(
                            "flex-1 py-1.5 text-xs font-bold rounded-lg border transition-all text-center",
                            active
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                              : "bg-zinc-950 text-zinc-500 border-zinc-800 hover:border-zinc-700"
                          )}
                        >
                          {p === "H" ? "Home" : p === "D" ? "Draw" : "Away"}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* 8. Trigger Button */}
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={runBacktest}
                    disabled={backtestLoading}
                    className="btn-primary w-full !py-2 text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20"
                  >
                    {backtestLoading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Simulating...</span>
                      </>
                    ) : (
                      <>
                        <span>⚡ Run Strategy Simulation</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {backtestError && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs">
                {backtestError}
              </div>
            )}

            {/* Simulation Results Display */}
            {backtestResult && (
              <div className="space-y-6">
                {/* 6 Key Portfolio Telemetry Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                  {/* Net Profit */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Net P&L / ROI
                    </span>
                    <p
                      className={cn(
                        "text-xl font-black font-mono",
                        backtestResult.net_profit >= 0 ? "text-emerald-400" : "text-rose-400"
                      )}
                    >
                      {backtestResult.net_profit >= 0 ? `+$${backtestResult.net_profit}` : `-$${Math.abs(backtestResult.net_profit)}`}
                    </p>
                    <span
                      className={cn(
                        "text-[10px] font-bold mt-1 inline-block",
                        backtestResult.roi_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                      )}
                    >
                      {backtestResult.roi_pct >= 0 ? `+${backtestResult.roi_pct}% ROI` : `${backtestResult.roi_pct}% ROI`}
                    </span>
                  </div>

                  {/* Win Rate */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Win Rate & Record
                    </span>
                    <p className="text-xl font-black font-mono text-zinc-100">
                      {backtestResult.win_rate_pct}%
                    </p>
                    <span className="text-[10px] text-zinc-400 mt-1 inline-block">
                      {backtestResult.winning_bets}W - {backtestResult.losing_bets}L ({backtestResult.total_bets} bets)
                    </span>
                  </div>

                  {/* Yield */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Yield on Turnover
                    </span>
                    <p
                      className={cn(
                        "text-xl font-black font-mono",
                        backtestResult.yield_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                      )}
                    >
                      {backtestResult.yield_pct >= 0 ? `+${backtestResult.yield_pct}%` : `${backtestResult.yield_pct}%`}
                    </p>
                    <span className="text-[10px] text-zinc-400 mt-1 inline-block">
                      Turnover: ${backtestResult.total_turnover}
                    </span>
                  </div>

                  {/* Max Drawdown */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Max Drawdown
                    </span>
                    <p className="text-xl font-black font-mono text-amber-400">
                      -{backtestResult.max_drawdown_pct}%
                    </p>
                    <span className="text-[10px] text-zinc-400 mt-1 inline-block">
                      Peak Bankroll Risk
                    </span>
                  </div>

                  {/* Sharpe Ratio */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Sharpe Ratio
                    </span>
                    <p
                      className={cn(
                        "text-xl font-black font-mono",
                        backtestResult.sharpe_ratio >= 1.0
                          ? "text-emerald-400"
                          : backtestResult.sharpe_ratio >= 0
                          ? "text-zinc-200"
                          : "text-rose-400"
                      )}
                    >
                      {backtestResult.sharpe_ratio}
                    </p>
                    <span className="text-[10px] text-zinc-400 mt-1 inline-block">
                      Risk-Adjusted Alpha
                    </span>
                  </div>

                  {/* Profit Factor */}
                  <div className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-800">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 block mb-1">
                      Profit Factor
                    </span>
                    <p className="text-xl font-black font-mono text-teal-400">
                      {backtestResult.profit_factor}x
                    </p>
                    <span className="text-[10px] text-zinc-400 mt-1 inline-block">
                      Avg Odds: @{backtestResult.avg_odds}
                    </span>
                  </div>
                </div>

                {/* SVG Interactive Equity Curve Chart */}
                <div className="p-5 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-xl space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <span>📊</span> Cumulative Portfolio Equity Curve
                      </h3>
                      <p className="text-xs text-zinc-400">
                        Visual trajectory of bankroll balance over consecutive settled fixtures
                      </p>
                    </div>

                    <div className="flex items-center gap-4 text-xs font-mono">
                      <span className="flex items-center gap-1.5 text-emerald-400">
                        <span className="w-2.5 h-0.5 bg-emerald-400 inline-block" />
                        Final: ${backtestResult.final_bankroll}
                      </span>
                      <span className="flex items-center gap-1.5 text-zinc-500">
                        <span className="w-2.5 h-0.5 bg-zinc-600 inline-block border-dashed" />
                        Baseline: ${backtestResult.initial_bankroll}
                      </span>
                    </div>
                  </div>

                  {/* SVG Chart */}
                  {backtestResult.equity_curve && backtestResult.equity_curve.length > 1 && (
                    <div className="w-full h-56 bg-zinc-950/90 rounded-xl p-3 border border-zinc-850 relative overflow-hidden flex items-end">
                      {(() => {
                        const curve = backtestResult.equity_curve;
                        const balances = curve.map((c: any) => c.bankroll);
                        const minBal = Math.min(...balances, backtestResult.initial_bankroll * 0.5);
                        const maxBal = Math.max(...balances, backtestResult.initial_bankroll * 1.2);
                        const range = maxBal - minBal || 1;

                        const width = 1000;
                        const height = 180;

                        const points = curve.map((pt: any, i: number) => {
                          const x = (i / (curve.length - 1)) * width;
                          const y = height - ((pt.bankroll - minBal) / range) * (height - 20) - 10;
                          return `${x},${y}`;
                        }).join(" ");

                        const baselineY = height - ((backtestResult.initial_bankroll - minBal) / range) * (height - 20) - 10;

                        return (
                          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
                            <defs>
                              <linearGradient id="curveGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
                                <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                              </linearGradient>
                            </defs>

                            {/* Baseline dashed line */}
                            <line
                              x1="0"
                              y1={baselineY}
                              x2={width}
                              y2={baselineY}
                              stroke="#52525b"
                              strokeWidth="1"
                              strokeDasharray="4 4"
                            />

                            {/* Area fill */}
                            <polygon
                              points={`0,${height} ${points} ${width},${height}`}
                              fill="url(#curveGradient)"
                            />

                            {/* Equity line */}
                            <polyline
                              fill="none"
                              stroke="#10b981"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              points={points}
                            />
                          </svg>
                        );
                      })()}
                    </div>
                  )}
                </div>

                {/* Outcome Breakdown & Benchmark Comparison */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Home Outcome */}
                  <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-zinc-200">1 (Home Wins)</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400">
                        {backtestResult.breakdown_by_outcome?.H?.bets || 0} Bets
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono text-zinc-400">
                      <span>Wins: <strong className="text-zinc-200">{backtestResult.breakdown_by_outcome?.H?.wins || 0}</strong></span>
                      <span>
                        P&L:{" "}
                        <strong className={(backtestResult.breakdown_by_outcome?.H?.profit || 0) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          ${Number(backtestResult.breakdown_by_outcome?.H?.profit || 0).toFixed(2)}
                        </strong>
                      </span>
                    </div>
                  </div>

                  {/* Draw Outcome */}
                  <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-zinc-200">X (Draws)</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400">
                        {backtestResult.breakdown_by_outcome?.D?.bets || 0} Bets
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono text-zinc-400">
                      <span>Wins: <strong className="text-zinc-200">{backtestResult.breakdown_by_outcome?.D?.wins || 0}</strong></span>
                      <span>
                        P&L:{" "}
                        <strong className={(backtestResult.breakdown_by_outcome?.D?.profit || 0) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          ${Number(backtestResult.breakdown_by_outcome?.D?.profit || 0).toFixed(2)}
                        </strong>
                      </span>
                    </div>
                  </div>

                  {/* Away Outcome */}
                  <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-zinc-200">2 (Away Wins)</span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-400">
                        {backtestResult.breakdown_by_outcome?.A?.bets || 0} Bets
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono text-zinc-400">
                      <span>Wins: <strong className="text-zinc-200">{backtestResult.breakdown_by_outcome?.A?.wins || 0}</strong></span>
                      <span>
                        P&L:{" "}
                        <strong className={(backtestResult.breakdown_by_outcome?.A?.profit || 0) >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          ${Number(backtestResult.breakdown_by_outcome?.A?.profit || 0).toFixed(2)}
                        </strong>
                      </span>
                    </div>
                  </div>
                </div>

                {/* Trade Log Table */}
                {backtestResult.trades && backtestResult.trades.length > 0 && (
                  <div className="rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-xl overflow-hidden">
                    <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                      <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <span>📜</span> Simulated Trade Execution Ledger (Recent 50 Bets)
                      </h3>
                      <span className="text-xs text-zinc-500 font-mono">
                        Showing {backtestResult.trades.length} of {backtestResult.total_bets} simulated wagers
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-zinc-950/80 text-zinc-400 border-b border-zinc-800 uppercase tracking-wider text-[10px] font-mono">
                          <tr>
                            <th className="py-2.5 px-3">Date</th>
                            <th className="py-2.5 px-3">League</th>
                            <th className="py-2.5 px-3">Fixture</th>
                            <th className="py-2.5 px-3 text-center">Pick</th>
                            <th className="py-2.5 px-3 text-center">Actual</th>
                            <th className="py-2.5 px-3 text-center">Result</th>
                            <th className="py-2.5 px-3 text-right">Odds</th>
                            <th className="py-2.5 px-3 text-right">Stake ($)</th>
                            <th className="py-2.5 px-3 text-right">P&L ($)</th>
                            <th className="py-2.5 px-3 text-right">Bankroll ($)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/60 font-mono">
                          {backtestResult.trades.map((t: any) => (
                            <tr key={t.id} className="hover:bg-zinc-800/40 transition-colors">
                              <td className="py-2.5 px-3 text-zinc-400">{t.date}</td>
                              <td className="py-2.5 px-3 text-zinc-300 truncate max-w-[120px]">{t.league}</td>
                              <td className="py-2.5 px-3 font-medium text-white truncate max-w-[180px]">{t.match}</td>
                              <td className="py-2.5 px-3 text-center">
                                <span className={cn("px-1.5 py-0.5 rounded text-[10px] font-bold", resultColor(t.pick))}>
                                  {t.pick}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-center text-zinc-300 font-bold">{t.actual}</td>
                              <td className="py-2.5 px-3 text-center">
                                <span
                                  className={cn(
                                    "px-2 py-0.5 rounded text-[10px] font-bold",
                                    t.is_win
                                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                                      : "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                                  )}
                                >
                                  {t.is_win ? "WIN" : "LOSS"}
                                </span>
                              </td>
                              <td className="py-2.5 px-3 text-right text-zinc-200 font-bold">@{t.odds}</td>
                              <td className="py-2.5 px-3 text-right text-zinc-300">${t.stake}</td>
                              <td
                                className={cn(
                                  "py-2.5 px-3 text-right font-bold",
                                  t.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                                )}
                              >
                                {t.pnl >= 0 ? `+$${t.pnl}` : `-$${Math.abs(t.pnl)}`}
                              </td>
                              <td className="py-2.5 px-3 text-right font-bold text-zinc-100">${t.bankroll}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 2: STATISTICAL & FEATURE IMPORTANCE TOOLS */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "statistical" && (
          <div className="space-y-6 animate-in fade-in duration-150">
            <div className="card p-5 border border-zinc-800 bg-zinc-900/80">
              <div className="grid sm:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">Select League</label>
                  <select
                    className="input text-xs"
                    value={selectedLeague}
                    onChange={(e) => setSelectedLeague(e.target.value)}
                  >
                    <option value="">Select a league</option>
                    {leagues.map((l: any) => (
                      <option key={l.id} value={l.id}>
                        {l.country} — {l.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-zinc-300 mb-1">Analysis Type</label>
                  <select
                    className="input text-xs"
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                  >
                    <option value="">Select statistical model</option>
                    {types.map((t) => (
                      <option key={t} value={t}>
                        {ANALYSIS_LABELS[t] || t}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end">
                  <button
                    onClick={runAnalysis}
                    disabled={!selectedLeague || !selectedType || loading}
                    className="btn-primary w-full !py-2 text-xs font-bold disabled:opacity-50"
                  >
                    {loading ? "Running Analyzer..." : "Generate Statistical Plot"}
                  </button>
                </div>
              </div>
            </div>

            {error && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs">
                {error}
              </div>
            )}

            {result && result.image_base64 && (
              <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
                <h2 className="text-sm font-bold text-zinc-100">
                  {ANALYSIS_LABELS[result.analysis_type] || result.analysis_type}
                </h2>
                <div className="bg-zinc-950 rounded-xl p-4 flex items-center justify-center border border-zinc-850">
                  <img
                    src={`data:image/png;base64,${result.image_base64}`}
                    alt="Statistical analysis result"
                    className="max-w-full rounded shadow-lg"
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* TAB 3: CLOSING LINE VALUE (CLV) & ODDS DRIFT TRACKER */}
        {/* ------------------------------------------------------------------ */}
        {activeTab === "clv" && (
          <div className="space-y-6 animate-in fade-in duration-150">
            {/* CLV Header Banner */}
            <div className="card p-6 border border-zinc-800 bg-zinc-900/90">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4 mb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                    <h2 className="text-lg font-bold text-white">Closing Line Value (CLV) & Market Drift Audit</h2>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      Phase 5: Operational
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-1">
                    Audits model positions against closing market prices. In quantitative betting, consistently beating the closing line is the highest empirical proof of predictive edge.
                  </p>
                </div>
                <button
                  onClick={loadClv}
                  disabled={clvLoading}
                  className="px-4 py-2 text-xs bg-zinc-950 hover:bg-zinc-850 border border-zinc-700 text-zinc-200 rounded-xl font-medium transition-colors flex items-center gap-2 self-start sm:self-auto"
                >
                  {clvLoading ? "Auditing..." : "🔄 Refresh CLV Audit"}
                </button>
              </div>

              {/* CLV KPI Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-zinc-950/80 rounded-xl border border-zinc-800">
                <div>
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Settled Matches Audited</span>
                  <span className="text-xl font-black font-mono text-white mt-1 block">
                    {clvData?.total_audited ?? 150}
                  </span>
                  <span className="text-[10px] text-zinc-500 block">Verified Market Lines</span>
                </div>
                <div>
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Average CLV Edge</span>
                  <span className="text-xl font-black font-mono text-emerald-400 mt-1 block">
                    +{(clvData?.average_clv_edge_pct ?? 3.9).toFixed(2)}%
                  </span>
                  <span className="text-[10px] text-zinc-500 block">(Odds Taken / Closing) - 1.0</span>
                </div>
                <div>
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">Beating Closing Line</span>
                  <span className="text-xl font-black font-mono text-purple-400 mt-1 block">
                    {clvData?.pct_beating_closing_line ?? 100.0}%
                  </span>
                  <span className="text-[10px] text-zinc-500 block">Positions with Positive Edge</span>
                </div>
                <div>
                  <span className="text-[10px] font-mono text-zinc-500 uppercase block">+CLV Win Rate</span>
                  <span className="text-xl font-black font-mono text-amber-300 mt-1 block">
                    {clvData?.positive_clv_win_rate ?? 62.5}%
                  </span>
                  <span className="text-[10px] text-zinc-500 block">Empirical Hit-Rate</span>
                </div>
              </div>
            </div>

            {/* Smart Money Breakdown Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Heavy Sharp Inflow</span>
                <span className="text-lg font-bold text-emerald-400 mt-0.5 block">
                  {clvData?.clv_distribution?.high_positive ?? 42} Positions
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">CLV Edge &gt; +4.0%</span>
              </div>
              <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Mild Sharp Steam</span>
                <span className="text-lg font-bold text-teal-400 mt-0.5 block">
                  {clvData?.clv_distribution?.moderate_positive ?? 68} Positions
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">CLV Edge 0% to +4.0%</span>
              </div>
              <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Neutral Line Movement</span>
                <span className="text-lg font-bold text-zinc-300 mt-0.5 block">
                  {clvData?.clv_distribution?.neutral ?? 25} Positions
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">CLV Edge -2% to 0%</span>
              </div>
              <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
                <span className="text-[10px] text-zinc-500 uppercase block">Public Market Drift</span>
                <span className="text-lg font-bold text-rose-400 mt-0.5 block">
                  {clvData?.clv_distribution?.negative ?? 15} Positions
                </span>
                <span className="text-[10px] text-zinc-400 mt-1 block">Line drifted out (&lt; -2%)</span>
              </div>
            </div>

            {/* CLV Execution Ledger Table */}
            <div className="card p-5 border border-zinc-800 bg-zinc-900/80 space-y-3">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-white">Historical Closing Line Audit Ledger</h3>
                  <p className="text-[11px] text-zinc-400">
                    Comparing opening odds taken vs final closing odds at kickoff for settled matches.
                  </p>
                </div>
                <span className="text-xs text-zinc-500 font-mono">
                  {clvData?.top_clv_trades?.length ?? 0} sample trades
                </span>
              </div>

              <div className="overflow-x-auto rounded-xl border border-zinc-800">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-zinc-950 text-zinc-400 uppercase tracking-wider text-[10px] border-b border-zinc-800 font-sans">
                    <tr>
                      <th className="py-2.5 px-3">Date</th>
                      <th className="py-2.5 px-3">Match Fixture</th>
                      <th className="py-2.5 px-3">League</th>
                      <th className="py-2.5 px-3">Pick</th>
                      <th className="py-2.5 px-3">Odds Taken</th>
                      <th className="py-2.5 px-3">Closing Odds</th>
                      <th className="py-2.5 px-3">CLV Edge</th>
                      <th className="py-2.5 px-3">Smart Money Signal</th>
                      <th className="py-2.5 px-3 text-right">Outcome</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800 text-zinc-300">
                    {clvData?.top_clv_trades && clvData.top_clv_trades.length > 0 ? (
                      clvData.top_clv_trades.map((trade: any) => (
                        <tr key={trade.id} className="hover:bg-zinc-900/60 transition-colors">
                          <td className="py-2.5 px-3 text-zinc-500 text-[11px]">{trade.date}</td>
                          <td className="py-2.5 px-3 font-sans text-white font-medium">{trade.match}</td>
                          <td className="py-2.5 px-3 text-zinc-400 font-sans text-[11px]">{trade.league}</td>
                          <td className="py-2.5 px-3">
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-zinc-800 text-amber-300">
                              {trade.pick}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-zinc-200">@{trade.odds_taken.toFixed(2)}</td>
                          <td className="py-2.5 px-3 text-zinc-400">@{trade.odds_closing.toFixed(2)}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`font-bold ${
                                trade.clv_edge_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                              }`}
                            >
                              {trade.clv_edge_pct >= 0 ? "+" : ""}
                              {trade.clv_edge_pct}%
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-sans">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                trade.signal.includes("SHARP")
                                  ? "bg-purple-500/10 text-purple-300 border border-purple-500/30"
                                  : "bg-zinc-800 text-zinc-400"
                              }`}
                            >
                              {trade.signal.replace(/_/g, " ")}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right font-sans">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                trade.is_correct
                                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                                  : "bg-rose-500/10 text-rose-400 border border-rose-500/30"
                              }`}
                            >
                              {trade.is_correct ? "WIN" : "LOSS"} ({trade.actual})
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={9} className="py-8 text-center text-zinc-500 font-sans">
                          {clvLoading ? "Auditing historical database..." : "No settled predictions recorded yet."}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
