"use client";

import React, { useState } from "react";
import { betslipApi } from "@/lib/api";
import { useAuth } from "@/lib/auth";

interface MatchStats {
  possession?: { home: number; away: number };
  shots_total?: { home: number; away: number };
  shots_on_target?: { home: number; away: number };
  corners?: { home: number; away: number };
  fouls?: { home: number; away: number };
  yellow_cards?: { home: number; away: number };
  xg?: { home: number; away: number };
  dangerous_attacks?: { home: number; away: number };
}

interface LiveStatsPanelProps {
  stats?: MatchStats;
  homeTeam: string;
  awayTeam: string;
  liveOdds?: { home: number; draw: number; away: number };
  leagueName: string;
  currentMinute: number;
}

export default function LiveStatsPanel({
  stats,
  homeTeam,
  awayTeam,
  liveOdds = { home: 2.10, draw: 3.25, away: 3.40 },
  leagueName,
  currentMinute,
}: LiveStatsPanelProps) {
  const { token } = useAuth();
  const [addedSelection, setAddedSelection] = useState<string | null>(null);

  const posH = stats?.possession?.home ?? 50;
  const posA = stats?.possession?.away ?? 50;
  const xgH = stats?.xg?.home ?? 0.8;
  const xgA = stats?.xg?.away ?? 0.6;
  const shotsTotalH = stats?.shots_total?.home ?? 6;
  const shotsTotalA = stats?.shots_total?.away ?? 4;
  const shotsTargetH = stats?.shots_on_target?.home ?? 3;
  const shotsTargetA = stats?.shots_on_target?.away ?? 2;
  const cornersH = stats?.corners?.home ?? 3;
  const cornersA = stats?.corners?.away ?? 2;
  const foulsH = stats?.fouls?.home ?? 5;
  const foulsA = stats?.fouls?.away ?? 7;
  const yellowsH = stats?.yellow_cards?.home ?? 1;
  const yellowsA = stats?.yellow_cards?.away ?? 2;
  const attacksH = stats?.dangerous_attacks?.home ?? 28;
  const attacksA = stats?.dangerous_attacks?.away ?? 21;

  const handleAddBet = async (selection: string, odds: number, bookmaker: string) => {
    if (!token) {
      alert("Please sign in to add live in-play selections to your Bet Journal.");
      return;
    }
    try {
      await betslipApi.createSlip(
        {
          match_title: `${homeTeam} vs ${awayTeam}`,
          league_name: leagueName,
          selection: `[Live ${currentMinute}'] ${selection}`,
          odds_taken: odds,
          stake_amount: 100,
          bookmaker,
          notes: `In-play selection taken at minute ${currentMinute}' via Live Match Watch`,
        },
        token
      );
      setAddedSelection(selection);
      setTimeout(() => setAddedSelection(null), 3000);
    } catch (err) {
      alert("Could not add bet slip. Check connection.");
    }
  };

  return (
    <div className="space-y-4">
      {/* Dynamic In-Play Odds Card */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/90 p-4 shadow-xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base">⚡</span>
            <span className="text-xs font-bold uppercase tracking-wider text-zinc-300 font-mono">
              Live In-Play Odds (South Africa)
            </span>
          </div>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
            Real-Time Vig
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center font-mono">
          {/* Home In-play */}
          <button
            type="button"
            onClick={() => handleAddBet(`${homeTeam} to Win`, liveOdds.home, "Hollywoodbets")}
            className="p-2.5 rounded-xl border border-zinc-800 bg-zinc-950 hover:border-emerald-500/50 hover:bg-emerald-950/20 transition group text-left"
          >
            <div className="text-[10px] text-zinc-500 truncate group-hover:text-zinc-300">
              1 • {homeTeam}
            </div>
            <div className="text-base font-black text-emerald-400 mt-0.5">
              @{liveOdds.home.toFixed(2)}
            </div>
            <div className="text-[9px] text-zinc-500">Hollywoodbets</div>
          </button>

          {/* Draw In-play */}
          <button
            type="button"
            onClick={() => handleAddBet("Draw", liveOdds.draw, "Betway SA")}
            className="p-2.5 rounded-xl border border-zinc-800 bg-zinc-950 hover:border-amber-500/50 hover:bg-amber-950/20 transition group text-left"
          >
            <div className="text-[10px] text-zinc-500 truncate group-hover:text-zinc-300">
              X • Draw
            </div>
            <div className="text-base font-black text-amber-400 mt-0.5">
              @{liveOdds.draw.toFixed(2)}
            </div>
            <div className="text-[9px] text-zinc-500">Betway SA</div>
          </button>

          {/* Away In-play */}
          <button
            type="button"
            onClick={() => handleAddBet(`${awayTeam} to Win`, liveOdds.away, "Sportingbet")}
            className="p-2.5 rounded-xl border border-zinc-800 bg-zinc-950 hover:border-purple-500/50 hover:bg-purple-950/20 transition group text-left"
          >
            <div className="text-[10px] text-zinc-500 truncate group-hover:text-zinc-300">
              2 • {awayTeam}
            </div>
            <div className="text-base font-black text-purple-400 mt-0.5">
              @{liveOdds.away.toFixed(2)}
            </div>
            <div className="text-[9px] text-zinc-500">Sportingbet</div>
          </button>
        </div>

        {addedSelection && (
          <div className="p-2 rounded-lg bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs text-center font-mono animate-fade-in">
            ✓ Added &quot;{addedSelection}&quot; to your Bet Journal!
          </div>
        )}
      </div>

      {/* Live Match Statistics */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/90 p-4 shadow-xl space-y-3.5">
        <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-300 font-mono">
            Live Match Statistics
          </span>
          <span className="text-[11px] font-mono text-zinc-500">
            {homeTeam} vs {awayTeam}
          </span>
        </div>

        {/* Possession */}
        <div className="space-y-1 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-emerald-400 font-bold">{posH}%</span>
            <span className="text-zinc-400 uppercase text-[10px]">Possession</span>
            <span className="text-purple-400 font-bold">{posA}%</span>
          </div>
          <div className="w-full h-1.5 bg-zinc-950 rounded-full flex overflow-hidden">
            <div className="bg-emerald-500 transition-all duration-500" style={{ width: `${posH}%` }} />
            <div className="bg-purple-500 transition-all duration-500" style={{ width: `${posA}%` }} />
          </div>
        </div>

        {/* Expected Goals (xG) */}
        <div className="space-y-1 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-emerald-400 font-bold">{xgH.toFixed(2)}</span>
            <span className="text-zinc-400 uppercase text-[10px]">Expected Goals (xG)</span>
            <span className="text-purple-400 font-bold">{xgA.toFixed(2)}</span>
          </div>
          <div className="w-full h-1.5 bg-zinc-950 rounded-full flex overflow-hidden">
            <div className="bg-emerald-500 transition-all duration-500" style={{ width: `${(xgH / Math.max(0.1, xgH + xgA)) * 100}%` }} />
            <div className="bg-purple-500 transition-all duration-500" style={{ width: `${(xgA / Math.max(0.1, xgH + xgA)) * 100}%` }} />
          </div>
        </div>

        {/* Dangerous Attacks */}
        <div className="space-y-1 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-emerald-400 font-bold">{attacksH}</span>
            <span className="text-zinc-400 uppercase text-[10px]">Dangerous Attacks</span>
            <span className="text-purple-400 font-bold">{attacksA}</span>
          </div>
          <div className="w-full h-1.5 bg-zinc-950 rounded-full flex overflow-hidden">
            <div className="bg-emerald-500 transition-all duration-500" style={{ width: `${(attacksH / Math.max(1, attacksH + attacksA)) * 100}%` }} />
            <div className="bg-purple-500 transition-all duration-500" style={{ width: `${(attacksA / Math.max(1, attacksH + attacksA)) * 100}%` }} />
          </div>
        </div>

        {/* Stats Grid Matrix */}
        <div className="grid grid-cols-2 gap-2 pt-2 border-t border-zinc-800 text-xs font-mono">
          <div className="p-2 rounded-lg bg-zinc-950 flex items-center justify-between">
            <span className="text-zinc-400">Total Shots:</span>
            <span className="font-bold text-white">{shotsTotalH} - {shotsTotalA}</span>
          </div>
          <div className="p-2 rounded-lg bg-zinc-950 flex items-center justify-between">
            <span className="text-zinc-400">On Target:</span>
            <span className="font-bold text-white">{shotsTargetH} - {shotsTargetA}</span>
          </div>
          <div className="p-2 rounded-lg bg-zinc-950 flex items-center justify-between">
            <span className="text-zinc-400">Corners:</span>
            <span className="font-bold text-white">{cornersH} - {cornersA}</span>
          </div>
          <div className="p-2 rounded-lg bg-zinc-950 flex items-center justify-between">
            <span className="text-zinc-400">Fouls:</span>
            <span className="font-bold text-white">{foulsH} - {foulsA}</span>
          </div>
          <div className="p-2 rounded-lg bg-zinc-950 flex items-center justify-between col-span-2">
            <span className="text-zinc-400">Yellow Cards:</span>
            <span className="font-bold text-amber-400">{yellowsH} - {yellowsA}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
