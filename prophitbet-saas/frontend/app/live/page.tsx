"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import DashboardLayout from "@/components/DashboardLayout";
import { liveMatchApi } from "@/lib/api";
import { cleanLeagueName, formatSASTDateTime } from "@/lib/utils";

interface Match {
  id: string;
  home_team: string;
  away_team: string;
  league_name: string;
  match_date: string | null;
  status: string;
  score_home: number | null;
  score_away: number | null;
}

export default function LiveMatchPage() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    liveMatchApi.getMatches().then((rows) => {
      if (active) setMatches(Array.isArray(rows) ? rows : []);
    }).catch((err) => {
      if (active) setError(err.message || "Unable to load matches");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Match Watch</h1>
        <div className="card border-amber-700 text-amber-200" role="status">
          Live commentary, in-play scores and odds are unavailable: no verified live-event feed is connected.
          The list below contains scheduled matches and verified final results, not a live simulation.
        </div>
        <Link href="/fixtures" className="text-emerald-400 underline">View all sourced fixtures</Link>
        {loading && <p role="status">Loading matches…</p>}
        {error && <p role="alert" className="text-red-400">{error}</p>}
        {!loading && !error && matches.length === 0 && <p>No verified matches available.</p>}
        <div className="grid gap-3 md:grid-cols-2">
          {matches.map((match) => (
            <article key={match.id} className="card space-y-2">
              <p className="text-sm text-zinc-400">{cleanLeagueName(match.league_name)} · {match.status}</p>
              <h2 className="font-semibold">{match.home_team} vs {match.away_team}</h2>
              <p>{formatSASTDateTime(match.match_date)}</p>
              {match.status === "FINISHED" && match.score_home !== null && match.score_away !== null
                ? <p>Final score: {match.score_home} – {match.score_away}</p>
                : <p className="text-zinc-400">Score not available</p>}
            </article>
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
