"use client";

import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { betslipApi } from "@/lib/api";
import Link from "next/link";

interface Slip {
  id: string;
  prediction_id: string | null;
  match_title: string;
  league_name: string;
  selection: string;
  odds_taken: number;
  stake_amount: number;
  status: "PENDING" | "WON" | "LOST" | "VOID";
  pnl: number;
  closing_odds: number | null;
  clv_edge_pct: number | null;
  notes: string | null;
  created_at: string;
  settled_at: string | null;
  bookmaker?: string | null;
  odds_origin?: string;
  legs?: { match_title: string; selection: string; odds_taken: number; status: string }[] | null;
}

interface PortfolioSummary {
  total_bets: number;
  pending_bets: number;
  settled_bets: number;
  total_wagered: number;
  total_pnl: number;
  roi_pct: number;
  win_rate_pct: number;
  avg_clv_edge_pct: number;
}

export default function BetSlipJournalPage() {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [slips, setSlips] = useState<Slip[]>([]);
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [autoSettling, setAutoSettling] = useState(false);
  const [autoSettleMsg, setAutoSettleMsg] = useState<string | null>(null);

  // New Bet Modal
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newMatch, setNewMatch] = useState("");
  const [newLeague, setNewLeague] = useState("Premier League");
  const [newSelection, setNewSelection] = useState("H");
  const [newOdds, setNewOdds] = useState("");
  const [newStake, setNewStake] = useState("");
  const [newNotes, setNewNotes] = useState("");
  const [saving, setSaving] = useState(false);

  const loadSlips = async () => {
    if (!token) return;
    try {
      setLoading(true);
      const res = await betslipApi.getSlips({ status: statusFilter }, token);
      setSummary(res.portfolio_summary);
      setSlips(res.slips || []);
    } catch (err) {
      console.error("Error loading bet slips:", err);
      setAutoSettleMsg("Could not load the journal. Please retry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSlips();
  }, [token, statusFilter]);

  const handleCreateSlip = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;
    try {
      setSaving(true);
      await betslipApi.createSlip(
        {
          match_title: newMatch,
          league_name: newLeague,
          selection: newSelection,
          odds_taken: parseFloat(newOdds),
          stake_amount: parseFloat(newStake),
          notes: newNotes,
        },
        token
      );
      setIsModalOpen(false);
      setNewNotes("");
      loadSlips();
    } catch (err) {
      console.error("Error creating slip:", err);
      setAutoSettleMsg(err instanceof Error ? err.message : "Could not save slip.");
    } finally {
      setSaving(false);
    }
  };

  const handleUpdateStatus = async (slipId: string, newStatus: string) => {
    if (!token) return;
    try {
      await betslipApi.updateSlip(slipId, { status: newStatus }, token);
      loadSlips();
    } catch (err) {
      console.error("Error updating slip:", err);
      setAutoSettleMsg("Could not update the slip. Please retry.");
    }
  };

  const handleDeleteSlip = async (slipId: string) => {
    if (!token) return;
    try {
      await betslipApi.deleteSlip(slipId, token);
      loadSlips();
    } catch (err) {
      console.error("Error deleting slip:", err);
      setAutoSettleMsg("Could not delete the slip. Please retry.");
    }
  };

  const handleAutoSettle = async () => {
    if (!token) return;
    try {
      setAutoSettling(true);
      const res = await betslipApi.autoSettle(token);
      setAutoSettleMsg(`Auto-settled ${res.settled_slips_count} pending bets against verified match outcomes.`);
      setTimeout(() => setAutoSettleMsg(null), 4000);
      loadSlips();
    } catch (err) {
      setAutoSettleMsg("Error auto-settling bets.");
      setTimeout(() => setAutoSettleMsg(null), 4000);
    } finally {
      setAutoSettling(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="text-2xl">📋</span>
              <h1 className="text-2xl font-black tracking-tight text-white">Personal Bet Journal & Bankroll</h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ZAR journal
              </span>
            </div>
            <p className="text-xs text-zinc-400 mt-1">
              Record bets in South African rand. Manual prices are unverified. Auto-settlement requires a linked prediction with a verified result. No bets are placed here.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/sa-markets" className="text-xs text-emerald-400">Build a slip</Link>
            <button
              onClick={handleAutoSettle}
              disabled={autoSettling}
              className="px-3.5 py-2 text-xs bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-200 rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              {autoSettling ? "Settling..." : "⚡ Auto-Settle Bets"}
            </button>
            <button
              onClick={() => setIsModalOpen(true)}
              className="px-4 py-2 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold transition-colors shadow-lg shadow-emerald-900/30 flex items-center gap-1.5"
            >
              <span>+</span> Log New Bet
            </button>
          </div>
        </div>

        {autoSettleMsg && (
          <div className="p-3 bg-emerald-950/40 border border-emerald-500/30 rounded-xl text-xs text-emerald-300">
            {autoSettleMsg}
          </div>
        )}

        {/* Portfolio KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Total Net P&L</span>
            <span
              className={`text-xl font-black font-mono block mt-1 ${
                (summary?.total_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {(summary?.total_pnl || 0) >= 0 ? "+" : ""}R{summary?.total_pnl?.toFixed(2) || "0.00"}
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">Bankroll Realized</span>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Net ROI / Yield</span>
            <span
              className={`text-xl font-black font-mono block mt-1 ${
                (summary?.roi_pct || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {(summary?.roi_pct || 0) >= 0 ? "+" : ""}{summary?.roi_pct || 0.0}%
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">Total Profit / Turnover</span>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Win Rate</span>
            <span className="text-xl font-black font-mono text-white block mt-1">
              {summary?.win_rate_pct || 0.0}%
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">
              {summary?.settled_bets || 0} Settled Bets
            </span>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Avg CLV Edge</span>
            <span className="text-xl font-black font-mono text-purple-400 block mt-1">
              {(summary?.avg_clv_edge_pct ?? 0).toFixed(1)}%
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">Beating Closing Line</span>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Total Wagered</span>
            <span className="text-xl font-black font-mono text-zinc-200 block mt-1">
              R{summary?.total_wagered?.toFixed(2) || "0.00"}
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">Cumulative Volume</span>
          </div>

          <div className="card p-4 border border-zinc-800 bg-zinc-900/80">
            <span className="text-[10px] font-mono text-zinc-500 uppercase block">Active Bets</span>
            <span className="text-xl font-black font-mono text-amber-300 block mt-1">
              {summary?.pending_bets || 0}
            </span>
            <span className="text-[10px] text-zinc-500 mt-0.5 block">In Play / Pending</span>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
          <div className="flex items-center gap-1.5">
            {["ALL", "PENDING", "WON", "LOST", "VOID"].map((st) => (
              <button
                key={st}
                onClick={() => setStatusFilter(st)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  statusFilter === st
                    ? "bg-zinc-800 text-white border border-zinc-700"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                {st}
              </button>
            ))}
          </div>
          <span className="text-xs text-zinc-500 font-mono">{slips.length} records</span>
        </div>

        {/* Bet Slips Table */}
        <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900/70">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-950/80 text-zinc-400 uppercase tracking-wider text-[10px] border-b border-zinc-800">
              <tr>
                <th className="py-3 px-4">Match & League</th>
                <th className="py-3 px-3">Selection</th>
                <th className="py-3 px-3">Odds Taken</th>
                <th className="py-3 px-3">Closing Odds</th>
                <th className="py-3 px-3">CLV Edge</th>
                <th className="py-3 px-3">Stake</th>
                <th className="py-3 px-3">P&L (ZAR)</th>
                <th className="py-3 px-3">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800 font-mono">
              {slips.length > 0 ? (
                slips.map((slip) => (
                  <tr key={slip.id} className="hover:bg-zinc-900/60 transition-colors">
                    <td className="py-3 px-4 font-sans font-medium text-white">
                      <div>{slip.match_title}</div>
                      <div className="text-[10px] text-zinc-500 font-mono">{slip.league_name}</div>
                      <div className="text-[10px] text-zinc-400">{slip.bookmaker || "Bookmaker not recorded"} · {slip.odds_origin === "observed" ? "Observed quote (not a confirmed wager)" : "Manually entered / unverified odds"}</div>
                      {slip.legs?.map((leg, index) => <div key={index} className="text-[10px] text-zinc-400 mt-1">{leg.match_title}: {leg.selection} @{leg.odds_taken.toFixed(2)} · {leg.status}</div>)}
                    </td>
                    <td className="py-3 px-3">
                      <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-zinc-800 text-amber-300 border border-zinc-700">
                        {slip.selection}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-zinc-200">@{slip.odds_taken.toFixed(2)}</td>
                    <td className="py-3 px-3 text-zinc-400">
                      {slip.closing_odds ? `@${slip.closing_odds.toFixed(2)}` : "—"}
                    </td>
                    <td className="py-3 px-3">
                      {slip.clv_edge_pct !== null ? (
                        <span
                          className={`text-[11px] font-bold ${
                            slip.clv_edge_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {slip.clv_edge_pct >= 0 ? "+" : ""}
                          {slip.clv_edge_pct}%
                        </span>
                      ) : (
                        <span className="text-zinc-600">—</span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-zinc-200">R{slip.stake_amount.toFixed(2)}</td>
                    <td className="py-3 px-3">
                      <span
                        className={`font-bold ${
                          slip.pnl > 0 ? "text-emerald-400" : slip.pnl < 0 ? "text-rose-400" : "text-zinc-500"
                        }`}
                      >
                        {slip.pnl > 0 ? "+" : ""}
                        R{slip.pnl.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-sans">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          slip.status === "WON"
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                            : slip.status === "LOST"
                            ? "bg-rose-500/10 text-rose-400 border border-rose-500/30"
                            : "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                        }`}
                      >
                        {slip.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-sans">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={() => handleUpdateStatus(slip.id, slip.status === "PENDING" ? "VOID" : "PENDING")}
                          className="px-2 py-0.5 rounded text-[10px] bg-zinc-800 text-zinc-300"
                        >
                          {slip.status === "PENDING" ? "Void / refund" : "Reopen"}
                        </button>
                        {slip.status === "PENDING" && (
                          <>
                            <button
                              onClick={() => handleUpdateStatus(slip.id, "WON")}
                              className="px-2 py-0.5 rounded text-[10px] bg-emerald-600/30 hover:bg-emerald-600 text-emerald-300 transition-colors"
                            >
                              Won
                            </button>
                            <button
                              onClick={() => handleUpdateStatus(slip.id, "LOST")}
                              className="px-2 py-0.5 rounded text-[10px] bg-rose-600/30 hover:bg-rose-600 text-rose-300 transition-colors"
                            >
                              Lost
                            </button>
                          </>
                        )}
                        <button
                          onClick={() => handleDeleteSlip(slip.id)}
                          className="px-1.5 py-0.5 text-zinc-600 hover:text-rose-400 transition-colors text-xs"
                          title="Delete"
                        >
                          ✕
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-zinc-500 font-sans">
                    No bet slips found. Click "Log New Bet" above to begin tracking your positions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Modal: Log New Bet */}
        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="card p-6 border border-zinc-700 bg-zinc-900 w-full max-w-md space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                <h3 className="text-base font-bold text-white">Log Position to Journal</h3>
                <button onClick={() => setIsModalOpen(false)} className="text-zinc-500 hover:text-white">
                  ✕
                </button>
              </div>

              <form onSubmit={handleCreateSlip} className="space-y-3.5 text-xs">
                {autoSettleMsg && <p role="alert" className="text-amber-300">{autoSettleMsg}</p>}
                <div>
                  <label className="text-zinc-400 block mb-1">Match Fixture</label>
                  <input
                    type="text"
                    value={newMatch}
                    onChange={(e) => setNewMatch(e.target.value)}
                    className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-zinc-400 block mb-1">League</label>
                    <input
                      type="text"
                      value={newLeague}
                      onChange={(e) => setNewLeague(e.target.value)}
                      className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                    />
                  </div>
                  <div>
                    <label className="text-zinc-400 block mb-1">Selection</label>
                    <select
                      value={newSelection}
                      onChange={(e) => setNewSelection(e.target.value)}
                      className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                    >
                      <option value="H">Home Win (H)</option>
                      <option value="D">Draw (D)</option>
                      <option value="A">Away Win (A)</option>
                      <option value="Over 2.5">Over 2.5 Goals</option>
                      <option value="Under 2.5">Under 2.5 Goals</option>
                      {["BTTS Yes", "BTTS No", "1X", "12", "X2", "DNB 1", "DNB 2"].map(pick => <option key={pick}>{pick}</option>)}
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-zinc-400 block mb-1">Odds Taken</label>
                    <input
                      type="number"
                      step="0.01"
                      min="1.01"
                      value={newOdds}
                      onChange={(e) => setNewOdds(e.target.value)}
                      className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="text-zinc-400 block mb-1">Stake Amount (ZAR)</label>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      value={newStake}
                      onChange={(e) => setNewStake(e.target.value)}
                      className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label className="text-zinc-400 block mb-1">Conviction / Strategy Notes</label>
                  <input
                    type="text"
                    placeholder="e.g. Sharp EV +7.2%, Quarter-Kelly stake recommendation"
                    value={newNotes}
                    onChange={(e) => setNewNotes(e.target.value)}
                    className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                  />
                </div>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setIsModalOpen(false)}
                    className="px-4 py-2 bg-zinc-800 text-zinc-300 rounded-lg hover:bg-zinc-700"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={saving}
                    className="px-4 py-2 bg-emerald-600 text-white font-bold rounded-lg hover:bg-emerald-500"
                  >
                    {saving ? "Saving..." : "Save Bet"}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
