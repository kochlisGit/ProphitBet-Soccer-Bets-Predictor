"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { leaguesApi } from "@/lib/api";
import {
  formatDate,
  cleanLeagueName,
  getCountryFlag,
  getLeaguePriority,
  getCountryPriority,
  TOP_LEAGUES,
} from "@/lib/utils";

type League = {
  id: number;
  name: string;
  country: string;
  category: string;
  start_year: number;
  last_synced_at?: string | null;
  dataset_count?: number;
  total_rows?: number;
};

export default function LeaguesPage() {
  const { token } = useAuth();
  const [leagues, setLeagues] = useState<League[]>([]);
  const [filter, setFilter] = useState("");
  const [selectedCountry, setSelectedCountry] = useState<string>("ALL");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) return;
    leaguesApi
      .list(token)
      .then(setLeagues)
      .catch(() => setLeagues([]))
      .finally(() => setLoading(false));
  }, [token]);

  // Featured Top Competitions (Premier League, La Liga, Serie A, etc.)
  const featuredLeagues = useMemo(() => {
    const list: (League & { rank: number; flag: string })[] = [];
    for (const top of TOP_LEAGUES) {
      const match = leagues.find(
        (l) =>
          l.country.toLowerCase() === top.country.toLowerCase() &&
          (l.name.toLowerCase().replace(/[\s-_]/g, "") ===
            top.name.toLowerCase().replace(/[\s-_]/g, "") ||
            l.name.toLowerCase().includes(top.name.toLowerCase()))
      );
      if (match) {
        list.push({ ...match, rank: top.rank, flag: top.flag });
      }
    }
    return list.sort((a, b) => a.rank - b.rank);
  }, [leagues]);

  // Available countries with counts
  const countryCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const l of leagues) {
      map.set(l.country, (map.get(l.country) || 0) + 1);
    }
    const list = [...map.entries()].map(([country, count]) => ({
      country,
      count,
      flag: getCountryFlag(country),
      priority: getCountryPriority(country),
    }));

    list.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return a.country.localeCompare(b.country);
    });

    return list;
  }, [leagues]);

  // Filtered leagues
  const filtered = useMemo(() => {
    return leagues.filter((l) => {
      // Country tab filter
      if (selectedCountry === "TOP5") {
        const top5 = ["England", "Spain", "Italy", "Germany", "France"];
        if (!top5.includes(l.country)) return false;
      } else if (selectedCountry !== "ALL" && l.country !== selectedCountry) {
        return false;
      }

      // Text search
      if (filter.trim()) {
        const q = filter.toLowerCase().trim();
        const matchesName = l.name.toLowerCase().includes(q);
        const matchesCountry = l.country.toLowerCase().includes(q);
        const matchesCategory = (l.category || "").toLowerCase().includes(q);
        if (!matchesName && !matchesCountry && !matchesCategory) return false;
      }

      return true;
    });
  }, [leagues, selectedCountry, filter]);

  // Group filtered by country, with priority countries first
  const groupedByCountry = useMemo(() => {
    const map = new Map<string, League[]>();
    for (const l of filtered) {
      if (!map.has(l.country)) map.set(l.country, []);
      map.get(l.country)!.push(l);
    }

    const groups = [...map.entries()].map(([country, items]) => {
      // Sort leagues within country by priority then name
      const sortedItems = [...items].sort((a, b) => {
        const pa = getLeaguePriority(a.name, a.country);
        const pb = getLeaguePriority(b.name, b.country);
        if (pa !== pb) return pa - pb;
        return a.name.localeCompare(b.name);
      });
      return {
        country,
        priority: getCountryPriority(country),
        flag: getCountryFlag(country),
        items: sortedItems,
      };
    });

    groups.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      return a.country.localeCompare(b.country);
    });

    return groups;
  }, [filtered]);

  return (
    <DashboardLayout>
      <div className="space-y-8">
      {/* Header & Main Navigation Bar */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-50">Leagues & Competitions</h1>
            <span className="badge bg-zinc-800 text-zinc-300 border-zinc-700 text-xs">
              {leagues.length} Available
            </span>
          </div>
          <p className="text-zinc-400 text-sm mt-1">
            Browse covered leagues, manage historical dataset syncs, and jump directly to predictions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/predictions"
            className="btn-primary !py-2 !px-4 text-sm inline-flex items-center gap-2 shadow-sm"
          >
            <span>🎯</span>
            <span>View All Predictions</span>
          </Link>
        </div>
      </div>

      {/* ⭐ TOP / POPULAR LEAGUES PINNED SECTION */}
      {featuredLeagues.length > 0 && selectedCountry === "ALL" && !filter && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-amber-400 text-base">⭐</span>
              <h2 className="text-lg font-bold text-zinc-100 tracking-tight">
                Top & Featured Competitions
              </h2>
              <span className="badge bg-amber-500/10 text-amber-300 border-amber-500/30 text-[10px]">
                High Activity
              </span>
            </div>
            <span className="text-xs text-zinc-500 hidden sm:inline">
              Europe&apos;s most active leagues with continuous predictions
            </span>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3.5">
            {featuredLeagues.map((league) => (
              <div
                key={league.id}
                className="card !p-4 bg-gradient-to-br from-zinc-800/80 to-zinc-900/90 border-zinc-700 hover:border-emerald-500/50 transition-all flex flex-col justify-between group shadow-sm"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{league.flag}</span>
                      <div>
                        <h3 className="text-sm font-semibold text-zinc-100 group-hover:text-emerald-400 transition">
                          {cleanLeagueName(league.name)}
                        </h3>
                        <p className="text-xs text-zinc-400">{league.country}</p>
                      </div>
                    </div>
                    <span className="badge bg-zinc-800 text-zinc-300 border-zinc-700 text-[10px]">
                      {league.category || "Top Flight"}
                    </span>
                  </div>

                  <div className="mt-3 pt-3 border-t border-zinc-800/80 flex items-center justify-between text-xs text-zinc-400">
                    <span>Since {league.start_year}</span>
                    {league.last_synced_at ? (
                      <span className="text-emerald-400/90 font-medium">Synced</span>
                    ) : (
                      <span className="text-zinc-500">Pending sync</span>
                    )}
                  </div>
                </div>

                {/* Direct Action Buttons */}
                <div className="mt-4 pt-2 flex items-center gap-2 border-t border-zinc-800/60">
                  <Link
                    href={`/predictions?league=${encodeURIComponent(league.name)}&leagueId=${league.id}`}
                    className="flex-1 text-center py-1.5 px-2 rounded-md text-xs font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition flex items-center justify-center gap-1"
                  >
                    <span>🎯</span>
                    <span>Predictions</span>
                  </Link>
                  <Link
                    href={`/leagues/${league.id}`}
                    className="flex-1 text-center py-1.5 px-2 rounded-md text-xs font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition"
                  >
                    Match Data
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filter and Search Toolbar */}
      <div className="card !p-4 bg-zinc-900/90 border-zinc-700/90 shadow-md space-y-3">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          {/* Live Search Input */}
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-zinc-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <input
              type="text"
              placeholder="Search by league name, country, or division..."
              className="input !pl-9 !py-2 text-sm bg-zinc-800/80 border-zinc-700 placeholder-zinc-500 w-full"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            {filter && (
              <button
                type="button"
                onClick={() => setFilter("")}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-zinc-400 hover:text-zinc-200 text-xs"
              >
                ✕ Clear
              </button>
            )}
          </div>

          <div className="text-xs text-zinc-400 whitespace-nowrap self-end md:self-auto">
            Showing <span className="text-zinc-200 font-semibold">{filtered.length}</span> of {leagues.length} leagues
          </div>
        </div>

        {/* Region & Country Filter Pills */}
        <div className="pt-2 border-t border-zinc-800 flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-zinc-400 font-medium mr-1">Region:</span>

          <button
            type="button"
            onClick={() => setSelectedCountry("ALL")}
            className={`px-3 py-1 text-xs rounded-full border transition ${
              selectedCountry === "ALL"
                ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            All Countries
          </button>

          <button
            type="button"
            onClick={() => setSelectedCountry("TOP5")}
            className={`px-3 py-1 text-xs rounded-full border transition ${
              selectedCountry === "TOP5"
                ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                : "bg-zinc-800/60 border-zinc-700/60 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            ⭐ Top 5 Nations
          </button>

          {countryCounts.slice(0, 8).map((c) => {
            const active = selectedCountry === c.country;
            return (
              <button
                key={c.country}
                type="button"
                onClick={() => setSelectedCountry(active ? "ALL" : c.country)}
                className={`px-2.5 py-1 text-xs rounded-full border transition inline-flex items-center gap-1.5 ${
                  active
                    ? "bg-emerald-600/20 border-emerald-500 text-emerald-300 font-semibold"
                    : "bg-zinc-800/60 border-zinc-700/60 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                }`}
              >
                <span>{c.flag}</span>
                <span>{c.country}</span>
                <span className="text-[10px] opacity-70 px-1 rounded bg-zinc-700/60">
                  {c.count}
                </span>
              </button>
            );
          })}

          {countryCounts.length > 8 && (
            <select
              value={
                countryCounts.slice(8).some((c) => c.country === selectedCountry)
                  ? selectedCountry
                  : ""
              }
              onChange={(e) => setSelectedCountry(e.target.value || "ALL")}
              className="input !py-1 !px-2 text-xs bg-zinc-800/90 border-zinc-700 text-zinc-300 !w-auto rounded-full"
              aria-label="More countries"
            >
              <option value="">Other nations ({countryCounts.length - 8})...</option>
              {countryCounts.slice(8).map((c) => (
                <option key={c.country} value={c.country}>
                  {c.country} ({c.count})
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Main Directory Grouped by Country */}
      {loading ? (
        <div className="card text-center py-16 text-zinc-400">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500 mb-3" />
          <p>Loading leagues and divisions...</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="card text-center py-16 text-zinc-400 space-y-3">
          <p className="text-base text-zinc-300 font-medium">No matching leagues found</p>
          <p className="text-sm text-zinc-500">
            Try resetting your search query or country filter.
          </p>
          <button
            type="button"
            onClick={() => {
              setFilter("");
              setSelectedCountry("ALL");
            }}
            className="btn-secondary !py-2 !px-4 text-xs inline-block mt-2"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <div className="space-y-8">
          {groupedByCountry.map((group) => (
            <div key={group.country} className="space-y-3">
              <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
                <span className="text-xl">{group.flag}</span>
                <h2 className="text-base font-bold text-zinc-100">{group.country}</h2>
                <span className="badge bg-zinc-800 text-zinc-400 border-zinc-700 text-xs font-normal">
                  {group.items.length} {group.items.length === 1 ? "competition" : "competitions"}
                </span>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
                {group.items.map((league) => (
                  <div
                    key={league.id}
                    className="card !p-4 bg-zinc-800/40 hover:bg-zinc-800/70 border-zinc-700/80 hover:border-zinc-600 transition flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="text-sm font-semibold text-zinc-100">
                          {cleanLeagueName(league.name)}
                        </h3>
                        <span className="badge bg-zinc-700/60 text-zinc-300 border-zinc-600/60 text-[10px]">
                          {league.category || "League"}
                        </span>
                      </div>

                      <div className="mt-2.5 flex items-center gap-3 text-xs text-zinc-400">
                        <span>Since {league.start_year}</span>
                        {league.last_synced_at && (
                          <span className="text-emerald-400/90">
                            Synced {formatDate(league.last_synced_at)}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Action shortcuts */}
                    <div className="mt-3.5 pt-2.5 border-t border-zinc-800 flex items-center gap-2">
                      <Link
                        href={`/predictions?league=${encodeURIComponent(league.name)}&leagueId=${league.id}`}
                        className="flex-1 text-center py-1.5 px-2 rounded-md text-xs font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition flex items-center justify-center gap-1"
                      >
                        <span>🎯</span>
                        <span>Predictions</span>
                      </Link>
                      <Link
                        href={`/leagues/${league.id}`}
                        className="flex-1 text-center py-1.5 px-2 rounded-md text-xs font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700 transition"
                      >
                        Match Data
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      </div>
    </DashboardLayout>
  );
}
