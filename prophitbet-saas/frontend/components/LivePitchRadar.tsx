"use client";

import React, { useMemo } from "react";

interface LivePitchRadarProps {
  ballX: number; // 0 (left goal) to 100 (right goal)
  ballY: number; // 0 (top touchline) to 100 (bottom touchline)
  homeTeam: string;
  awayTeam: string;
  currentEvent?: {
    type: string;
    badge: string;
    team: string;
    minute: number;
    description: string;
  } | null;
  momentumHome: number; // 0 to 100
  momentumAway: number;
}

export default function LivePitchRadar({
  ballX = 50,
  ballY = 50,
  homeTeam,
  awayTeam,
  currentEvent,
  momentumHome = 50,
  momentumAway = 50,
}: LivePitchRadarProps) {
  // Determine attacking zone description
  const zoneDescription = useMemo(() => {
    if (ballX > 85) return `${homeTeam} in the penalty box!`;
    if (ballX > 65) return `${homeTeam} attacking final third`;
    if (ballX < 15) return `${awayTeam} in the penalty box!`;
    if (ballX < 35) return `${awayTeam} attacking final third`;
    return "Contested in midfield";
  }, [ballX, homeTeam, awayTeam]);

  const isGoal = currentEvent?.type === "GOAL";
  const isAttack = currentEvent?.type === "DANGEROUS_ATTACK" || ballX > 75 || ballX < 25;

  return (
    <div className="relative rounded-2xl overflow-hidden border border-emerald-500/20 bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 shadow-2xl p-4 sm:p-5 flex flex-col justify-between">
      {/* Pitch Header / State Indicator */}
      <div className="flex items-center justify-between gap-3 mb-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </span>
          <span className="font-mono font-bold tracking-wider uppercase text-emerald-400">
            2D Tactical Radar
          </span>
        </div>

        <div className="text-zinc-400 font-mono text-[11px] truncate max-w-[220px]">
          <span className="text-zinc-300 font-semibold">{zoneDescription}</span>
        </div>

        {/* Momentum Indicator */}
        <div className="hidden sm:flex items-center gap-2 text-[10px] font-mono">
          <span className="text-emerald-400">{homeTeam}: {momentumHome}%</span>
          <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden flex">
            <div className="bg-emerald-500 transition-all duration-500" style={{ width: `${momentumHome}%` }} />
            <div className="bg-purple-500 transition-all duration-500" style={{ width: `${momentumAway}%` }} />
          </div>
          <span className="text-purple-400">{awayTeam}: {momentumAway}%</span>
        </div>
      </div>

      {/* 2D Grass Pitch Arena */}
      <div className="relative w-full aspect-[2/1] rounded-xl overflow-hidden border-2 border-emerald-600/40 shadow-inner bg-[#13331b] select-none">
        {/* Grass Alternating Stripes Texture */}
        <div className="absolute inset-0 flex">
          {Array.from({ length: 12 }).map((_, idx) => (
            <div
              key={idx}
              className={`flex-1 h-full ${
                idx % 2 === 0 ? "bg-emerald-900/30" : "bg-emerald-900/10"
              }`}
            />
          ))}
        </div>

        {/* Pitch Markings */}
        {/* Outer Boundary Line */}
        <div className="absolute inset-2 sm:inset-3 border border-emerald-400/40 rounded-sm pointer-events-none" />

        {/* Halfway Line */}
        <div className="absolute top-2 sm:top-3 bottom-2 sm:bottom-3 left-1/2 w-0.5 -translate-x-1/2 border-r border-emerald-400/40 pointer-events-none" />

        {/* Center Circle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 sm:w-24 md:w-28 aspect-square rounded-full border border-emerald-400/40 pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-emerald-400/60 pointer-events-none" />

        {/* Left Penalty Area (Away Goal) */}
        <div className="absolute top-1/2 left-2 sm:left-3 -translate-y-1/2 w-[16%] h-[58%] border-t border-r border-b border-emerald-400/40 pointer-events-none" />
        <div className="absolute top-1/2 left-2 sm:left-3 -translate-y-1/2 w-[6%] h-[28%] border-t border-r border-b border-emerald-400/40 pointer-events-none" />
        {/* Left Goal Mouth */}
        <div className="absolute top-1/2 left-0 -translate-y-1/2 w-2 h-[18%] border-t border-l border-b border-amber-400/60 bg-amber-400/20 pointer-events-none" />

        {/* Right Penalty Area (Home Goal) */}
        <div className="absolute top-1/2 right-2 sm:right-3 -translate-y-1/2 w-[16%] h-[58%] border-t border-l border-b border-emerald-400/40 pointer-events-none" />
        <div className="absolute top-1/2 right-2 sm:right-3 -translate-y-1/2 w-[6%] h-[28%] border-t border-l border-b border-emerald-400/40 pointer-events-none" />
        {/* Right Goal Mouth */}
        <div className="absolute top-1/2 right-0 -translate-y-1/2 w-2 h-[18%] border-t border-r border-b border-amber-400/60 bg-amber-400/20 pointer-events-none" />

        {/* Team Direction Labels on Turf */}
        <div className="absolute top-4 left-6 text-[10px] font-mono uppercase font-bold text-emerald-400/40 tracking-wider">
          ← {awayTeam} Defending
        </div>
        <div className="absolute top-4 right-6 text-[10px] font-mono uppercase font-bold text-emerald-400/40 tracking-wider text-right">
          {homeTeam} Defending →
        </div>

        {/* Dangerous Attack Pulse Rings */}
        {isAttack && (
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-all duration-700 ease-out"
            style={{ left: `${ballX}%`, top: `${ballY}%` }}
          >
            <div className="w-16 h-16 rounded-full border-2 border-amber-400 animate-ping opacity-60" />
          </div>
        )}

        {/* Goal Explosion Effect */}
        {isGoal && (
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-all duration-500"
            style={{ left: `${ballX}%`, top: `${ballY}%` }}
          >
            <div className="w-24 h-24 rounded-full bg-emerald-400/30 animate-ping" />
            <div className="w-12 h-12 rounded-full bg-amber-400/40 animate-pulse" />
          </div>
        )}

        {/* The Ball */}
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2 z-20 transition-all duration-700 ease-out"
          style={{ left: `${Math.max(4, Math.min(96, ballX))}%`, top: `${Math.max(6, Math.min(94, ballY))}%` }}
        >
          <div className="relative group">
            {/* Glow Aura */}
            <div className="absolute -inset-1.5 rounded-full bg-amber-400/50 blur-sm animate-pulse" />
            {/* Ball Body */}
            <div className="relative w-4 h-4 rounded-full bg-white border-2 border-zinc-900 shadow-xl flex items-center justify-center">
              <div className="w-1.5 h-1.5 rounded-full bg-zinc-900" />
            </div>
            {/* Match Event Tag Float */}
            {currentEvent?.badge && (
              <div className="absolute bottom-5 left-1/2 -translate-x-1/2 whitespace-nowrap px-2 py-0.5 rounded-full text-[10px] font-black bg-zinc-900/90 text-amber-300 border border-amber-500/40 shadow-lg pointer-events-none animate-bounce">
                {currentEvent.badge}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Pitch Footer Banner */}
      <div className="mt-3 flex items-center justify-between text-xs text-zinc-400 font-mono">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
          <span>Home Attack: <strong>{homeTeam}</strong></span>
        </span>
        <span className="text-[11px] text-zinc-500">Live coordinates: [{ballX}m, {ballY}m]</span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-purple-400" />
          <span>Away Attack: <strong>{awayTeam}</strong></span>
        </span>
      </div>
    </div>
  );
}
