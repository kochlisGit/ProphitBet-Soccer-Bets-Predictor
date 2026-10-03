"use client";

import React, { useState, useEffect, useRef } from "react";

export interface CommentaryEvent {
  minute: number;
  period: string;
  type: string;
  team: string;
  badge: string;
  description: string;
  score_home: number;
  score_away: number;
}

interface LiveCommentaryFeedProps {
  events: CommentaryEvent[];
  currentMinute: number;
  homeTeam: string;
  awayTeam: string;
}

export default function LiveCommentaryFeed({
  events,
  currentMinute,
  homeTeam,
  awayTeam,
}: LiveCommentaryFeedProps) {
  const [filter, setFilter] = useState<"ALL" | "KEY" | "GOALS">("ALL");
  const [autoScroll, setAutoScroll] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [search, setSearch] = useState("");
  const feedEndRef = useRef<HTMLDivElement>(null);
  const lastSpokenMinuteRef = useRef<number>(-1);

  // Events filtered up to the current playback minute
  const revealedEvents = events.filter((e) => e.minute <= currentMinute);

  // Category and search filtering
  const displayEvents = revealedEvents.filter((e) => {
    if (filter === "GOALS" && e.type !== "GOAL") return false;
    if (filter === "KEY" && !["GOAL", "RED_CARD", "YELLOW_CARD", "SAVE", "VAR_CHECK"].includes(e.type)) {
      return false;
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      return e.description.toLowerCase().includes(q) || e.badge.toLowerCase().includes(q) || e.team.toLowerCase().includes(q);
    }
    return true;
  });

  // Auto-scroll on new events
  useEffect(() => {
    if (autoScroll && feedEndRef.current) {
      feedEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [revealedEvents.length, autoScroll]);

  // Text-To-Speech audio commentary
  useEffect(() => {
    if (!audioEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }
    const latestEvent = revealedEvents[revealedEvents.length - 1];
    if (latestEvent && latestEvent.minute > lastSpokenMinuteRef.current) {
      lastSpokenMinuteRef.current = latestEvent.minute;
      window.speechSynthesis.cancel(); // Stop prior speech
      const textToSpeak = `Minute ${latestEvent.minute}: ${latestEvent.description}`;
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }, [revealedEvents, audioEnabled]);

  const getEventStyle = (type: string) => {
    switch (type) {
      case "GOAL":
        return "border-emerald-500/50 bg-emerald-950/30 text-emerald-300";
      case "RED_CARD":
        return "border-rose-500/50 bg-rose-950/30 text-rose-300";
      case "YELLOW_CARD":
        return "border-amber-500/40 bg-amber-950/20 text-amber-300";
      case "SAVE":
        return "border-sky-500/40 bg-sky-950/20 text-sky-300";
      case "CORNER":
        return "border-indigo-500/30 bg-indigo-950/20 text-indigo-300";
      case "DANGEROUS_ATTACK":
        return "border-purple-500/30 bg-purple-950/20 text-purple-300";
      case "HALFTIME":
      case "FULLTIME":
        return "border-zinc-700 bg-zinc-800 text-zinc-200 font-bold";
      default:
        return "border-zinc-800 bg-zinc-900/60 text-zinc-300";
    }
  };

  return (
    <div className="flex flex-col h-[520px] rounded-2xl border border-zinc-800 bg-zinc-900/90 shadow-xl overflow-hidden">
      {/* Feed Controls Header */}
      <div className="p-3.5 border-b border-zinc-800 bg-zinc-950/90 flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🎙️</span>
          <span className="font-bold text-sm text-zinc-100">Live Commentary</span>
          <span className="text-xs px-2 py-0.5 rounded-full font-mono font-bold bg-zinc-800 text-zinc-400">
            {displayEvents.length} events
          </span>
        </div>

        {/* Audio Narration Toggle */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setAudioEnabled(!audioEnabled)}
            className={`px-2.5 py-1 text-xs font-semibold rounded-lg border transition-all flex items-center gap-1.5 ${
              audioEnabled
                ? "bg-emerald-600/30 border-emerald-500 text-emerald-300 shadow-sm"
                : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
            }`}
            title="Listen to live commentary using voice synthesis"
          >
            <span>{audioEnabled ? "🔊 Voice Active" : "🔇 Voice Off"}</span>
          </button>

          <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded border-zinc-700 bg-zinc-900 text-emerald-500 focus:ring-0"
            />
            <span>Auto-scroll</span>
          </label>
        </div>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="px-3.5 py-2 border-b border-zinc-800/80 bg-zinc-950/50 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setFilter("ALL")}
            className={`px-2.5 py-1 rounded-md font-medium transition ${
              filter === "ALL" ? "bg-zinc-800 text-white font-bold" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            All Action
          </button>
          <button
            onClick={() => setFilter("KEY")}
            className={`px-2.5 py-1 rounded-md font-medium transition ${
              filter === "KEY" ? "bg-amber-500/20 text-amber-300 border border-amber-500/30 font-bold" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            ⚡ Key Moments
          </button>
          <button
            onClick={() => setFilter("GOALS")}
            className={`px-2.5 py-1 rounded-md font-medium transition ${
              filter === "GOALS" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold" : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            ⚽ Goals Only
          </button>
        </div>

        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter moments..."
          className="bg-zinc-900 border border-zinc-800 rounded-md px-2.5 py-0.5 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-emerald-500 w-32 sm:w-40"
        />
      </div>

      {/* Scrollable Commentary Feed List */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-2.5 font-sans">
        {displayEvents.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-500 text-xs py-12 space-y-2">
            <span className="text-2xl">⏳</span>
            <p>Awaiting kickoff incidents or match events...</p>
          </div>
        ) : (
          displayEvents.map((ev, idx) => {
            const isLatest = idx === displayEvents.length - 1;
            return (
              <div
                key={idx}
                className={`p-3 rounded-xl border text-xs leading-relaxed transition-all shadow-sm ${getEventStyle(
                  ev.type
                )} ${isLatest ? "ring-1 ring-emerald-500/40" : ""}`}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-black text-xs px-2 py-0.5 rounded bg-zinc-950/70 border border-zinc-800">
                      {ev.minute}&apos;
                    </span>
                    <span className="font-bold text-[11px] tracking-wide uppercase">
                      {ev.badge}
                    </span>
                  </div>

                  <span className="font-mono text-[11px] font-bold text-zinc-400">
                    {ev.score_home} - {ev.score_away}
                  </span>
                </div>

                <p className="text-zinc-200 text-xs mt-1 font-normal">
                  {ev.description}
                </p>
              </div>
            );
          })
        )}
        <div ref={feedEndRef} />
      </div>
    </div>
  );
}
