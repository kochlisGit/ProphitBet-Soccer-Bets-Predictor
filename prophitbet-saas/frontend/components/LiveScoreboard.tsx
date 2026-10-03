"use client";

import React from "react";
import { getCountryFlag, cleanLeagueName } from "@/lib/utils";

interface LiveScoreboardProps {
  homeTeam: string;
  awayTeam: string;
  leagueName: string;
  country?: string;
  scoreHome: number;
  scoreAway: number;
  currentMinute: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  speed: number;
  onChangeSpeed: (speed: number) => void;
  onScrubMinute: (minute: number) => void;
  probHome: number; // 0 to 100
  probDraw: number;
  probAway: number;
  status?: string;
}

export default function LiveScoreboard({
  homeTeam,
  awayTeam,
  leagueName,
  country,
  scoreHome,
  scoreAway,
  currentMinute,
  isPlaying,
  onTogglePlay,
  speed,
  onChangeSpeed,
  onScrubMinute,
  probHome = 45,
  probDraw = 25,
  probAway = 30,
  status = "LIVE",
}: LiveScoreboardProps) {
  const flag = getCountryFlag(country || leagueName);
  const cleanLg = cleanLeagueName(leagueName, country);

  const getStatusDisplay = () => {
    if (currentMinute >= 90) return "FULL TIME";
    if (currentMinute === 45) return "HALF TIME";
    if (currentMinute === 0) return "UPCOMING";
    return `LIVE ${currentMinute}'`;
  };

  return (
    <div className="rounded-2xl border border-zinc-700/80 bg-gradient-to-b from-zinc-900 via-zinc-900/90 to-zinc-950 p-5 shadow-2xl space-y-5">
      {/* Competition & Status Pill Header */}
      <div className="flex items-center justify-between gap-3 text-xs border-b border-zinc-800 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-base">{flag}</span>
          <span className="font-semibold text-zinc-300">{cleanLg}</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${currentMinute >= 90 ? "bg-zinc-500" : "bg-rose-400"} opacity-75`} />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${currentMinute >= 90 ? "bg-zinc-500" : "bg-rose-500"}`} />
          </span>
          <span className={`px-2.5 py-0.5 rounded-full font-mono text-xs font-black tracking-wider ${
            currentMinute >= 90
              ? "bg-zinc-800 text-zinc-300 border border-zinc-700"
              : "bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse"
          }`}>
            {getStatusDisplay()}
          </span>
        </div>
      </div>

      {/* Main Scoreboard Display */}
      <div className="grid grid-cols-12 items-center gap-4 py-2">
        {/* Home Team */}
        <div className="col-span-5 text-right space-y-1">
          <div className="text-xl sm:text-2xl font-black text-white tracking-tight truncate">
            {homeTeam}
          </div>
          <div className="text-xs font-mono text-emerald-400 font-semibold">
            Home • Win Prob: {probHome}%
          </div>
        </div>

        {/* Live Score Counter */}
        <div className="col-span-2 flex flex-col items-center justify-center">
          <div className="flex items-center justify-center gap-2 font-mono font-black text-3xl sm:text-4xl text-white px-4 py-1.5 rounded-xl bg-zinc-950/80 border border-zinc-800 shadow-inner">
            <span className="text-emerald-400">{scoreHome}</span>
            <span className="text-zinc-600 text-xl font-normal">:</span>
            <span className="text-purple-400">{scoreAway}</span>
          </div>
          <div className="text-[11px] font-mono text-zinc-500 mt-1">
            Min {currentMinute}&apos; / 90&apos;
          </div>
        </div>

        {/* Away Team */}
        <div className="col-span-5 text-left space-y-1">
          <div className="text-xl sm:text-2xl font-black text-white tracking-tight truncate">
            {awayTeam}
          </div>
          <div className="text-xs font-mono text-purple-400 font-semibold">
            Away • Win Prob: {probAway}%
          </div>
        </div>
      </div>

      {/* Live Win Probability Bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px] font-mono font-semibold">
          <span className="text-emerald-400">{homeTeam} {probHome}%</span>
          <span className="text-amber-400">Draw {probDraw}%</span>
          <span className="text-purple-400">{awayTeam} {probAway}%</span>
        </div>
        <div className="w-full h-2 bg-zinc-950 rounded-full overflow-hidden flex shadow-inner">
          <div className="bg-emerald-500 transition-all duration-700" style={{ width: `${probHome}%` }} />
          <div className="bg-amber-500 transition-all duration-700" style={{ width: `${probDraw}%` }} />
          <div className="bg-purple-500 transition-all duration-700" style={{ width: `${probAway}%` }} />
        </div>
      </div>

      {/* Interactive Simulation Controls & Scrubber */}
      <div className="pt-3 border-t border-zinc-800/80 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 text-xs">
        {/* Play/Pause & Speed Buttons */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onTogglePlay}
            className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 shadow-sm ${
              isPlaying
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30"
                : "bg-emerald-600 text-white hover:bg-emerald-500 shadow-emerald-900/50"
            }`}
          >
            <span>{isPlaying ? "⏸️ Pause" : "▶️ Play Live"}</span>
          </button>

          {/* Speed Multipliers */}
          <div className="flex items-center rounded-lg border border-zinc-800 bg-zinc-950 p-0.5">
            {[1, 2, 5, 10].map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onChangeSpeed(s)}
                className={`px-2 py-1 text-[11px] font-mono font-bold rounded transition ${
                  speed === s
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => onScrubMinute(0)}
            className="px-2 py-1 text-[11px] font-mono text-zinc-500 hover:text-zinc-300 border border-zinc-800 rounded"
            title="Reset to minute 1"
          >
            ⏮ Reset
          </button>
          <button
            type="button"
            onClick={() => onScrubMinute(90)}
            className="px-2 py-1 text-[11px] font-mono text-zinc-500 hover:text-zinc-300 border border-zinc-800 rounded"
            title="Fast forward to Full Time"
          >
            ⏭ FT
          </button>
        </div>

        {/* Timeline Scrubber Slider */}
        <div className="flex items-center gap-2 flex-1 sm:max-w-xs">
          <span className="font-mono text-[11px] text-zinc-500 whitespace-nowrap">
            Min {currentMinute}&apos;
          </span>
          <input
            type="range"
            min={1}
            max={90}
            value={currentMinute}
            onChange={(e) => onScrubMinute(Number(e.target.value))}
            className="w-full accent-emerald-500 h-1.5 bg-zinc-800 rounded-lg cursor-pointer"
          />
          <span className="font-mono text-[11px] text-zinc-500">90&apos;</span>
        </div>
      </div>
    </div>
  );
}
