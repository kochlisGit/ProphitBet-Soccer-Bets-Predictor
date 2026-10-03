"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { leaguesApi } from "@/lib/api";

export default function LeagueDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { token, user } = useAuth();
  const [league, setLeague] = useState<any>(null);
  const [tableData, setTableData] = useState<any>(null);
  const [tableError, setTableError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token || !id) return;
    leaguesApi
      .get(Number(id), token)
      .then(setLeague)
      .catch(() => setLeague(null))
      .finally(() => setLoading(false));
  }, [token, id]);

  useEffect(() => {
    if (!token || !id) return;
    setTableError(null);
    leaguesApi
      .table(Number(id), page, token)
      .then((data) => {
        setTableData(data);
        setTableError(null);
      })
      .catch((err: any) => {
        setTableData(null);
        setTableError(err?.message || "Could not load match data");
      });
  }, [token, id, page]);

  return (
    <DashboardLayout>
      {loading ? (
        <div className="text-center py-20 text-gray-500">Loading...</div>
      ) : !league ? (
        <div className="text-center py-20 text-gray-500">League not found</div>
      ) : (
        <>
          {/* Breadcrumb Navigation */}
          <div className="flex items-center gap-2 text-xs text-zinc-400 mb-4">
            <Link href="/dashboard" className="hover:text-zinc-200 transition">Dashboard</Link>
            <span>/</span>
            <Link href="/leagues" className="hover:text-zinc-200 transition">Leagues</Link>
            <span>/</span>
            <span className="text-zinc-300">{league.country}</span>
            <span>/</span>
            <span className="text-emerald-400 font-medium">{league.name}</span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
            <div>
              <h1 className="anim-page-title text-2xl font-bold text-white">{league.name}</h1>
              <p className="anim-page-sub text-gray-400 mt-1">
                {league.country} &middot; {league.category} &middot; Since {league.start_year}
              </p>
            </div>
            <div className="flex items-center gap-2.5">
              <Link
                href={`/predictions?league=${encodeURIComponent(league.name)}&leagueId=${league.id}`}
                className="btn-primary !py-2 !px-4 text-sm inline-flex items-center gap-1.5 shadow-sm"
              >
                <span>🎯</span>
                <span>View {league.name} Predictions</span>
              </Link>
              <Link
                href="/fixtures"
                className="btn-secondary !py-2 !px-3.5 text-sm inline-flex items-center gap-1.5"
              >
                Fixtures
              </Link>
            </div>
          </div>

          <div className="anim-features-section grid sm:grid-cols-3 gap-4 mb-8">
            <div className="anim-feature-card card">
              <p className="text-sm text-gray-500">Datasets</p>
              <p className="text-2xl font-bold text-white mt-1">{league.dataset_count}</p>
            </div>
            <div className="anim-feature-card card">
              <p className="text-sm text-gray-500">Total Rows</p>
              <p className="text-2xl font-bold text-white mt-1">{league.total_rows?.toLocaleString()}</p>
            </div>
            <div className="anim-feature-card card">
              <p className="text-sm text-gray-500">Last Synced</p>
              <p className="text-lg font-bold text-white mt-1">
                {league.last_synced_at
                  ? new Date(league.last_synced_at).toLocaleDateString()
                  : "Never"}
              </p>
            </div>
          </div>

          {league.category === "public-json" && (
            <div className="card mb-6 text-sm text-gray-400">
              Champions League results from 2017 onward are sourced from FixtureDownload.
              Source links and score scope are included in the match table. Knockout scores may
              include extra time; only group/league-phase scores are used for 90-minute training
              and settlement. Historical results are not retrospective predictions.
            </div>
          )}
          {/* Data Table */}
          {(!league.dataset_count || league.dataset_count === 0) && (
            <div className="anim-reveal-section">
              <div className="anim-reveal-inner card border border-amber-900/50 bg-amber-950/20">
              <p className="text-amber-200/90 text-sm font-medium mb-2">No match data for this league yet</p>
              <p className="text-gray-400 text-sm mb-4">
                Datasets are created when the platform downloads and processes results from its configured providers, then stores them
                in object storage. Until that sync finishes for this league, tables, stats, and training stay empty.
              </p>
              {user?.is_admin ? (
                <Link href="/admin" className="text-sm btn-primary !py-2 !px-4 inline-block">
                  Open Admin → Sync League Data
                </Link>
              ) : (
                <p className="text-xs text-gray-500">Ask an administrator to run &quot;Sync League Data&quot; in the Admin panel.</p>
              )}
            </div>
            </div>
          )}

          {league.dataset_count > 0 && tableError && (
            <div className="anim-reveal-section">
              <div className="anim-reveal-inner card border border-red-900/40">
              <p className="text-red-300 text-sm font-medium mb-1">Could not load match table</p>
              <p className="text-gray-400 text-sm">{tableError}</p>
            </div>
            </div>
          )}

          {tableData && tableData.data && tableData.data.length > 0 && (
            <div className="anim-reveal-section">
              <div className="anim-reveal-inner card overflow-hidden">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white">Match Data</h2>
                <span className="text-sm text-gray-500">
                  {tableData.total} rows &middot; Page {page}
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-left text-gray-500 border-b border-gray-800">
                      {tableData.columns.slice(0, 12).map((col: string) => (
                        <th key={col} className="pb-2 pr-4 font-medium whitespace-nowrap">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800/50">
                    {tableData.data.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-gray-800/30">
                        {tableData.columns.slice(0, 12).map((col: string) => (
                          <td key={col} className="py-2 pr-4 text-gray-300 whitespace-nowrap">
                            {col === "Source" && typeof row[col] === "string" && row[col].startsWith("https://fixturedownload.com/") ? (
                              <a href={row[col]} target="_blank" rel="noreferrer" className="text-emerald-400 underline">Source</a>
                            ) : typeof row[col] === "number" ? (Number.isInteger(row[col]) ? row[col] : row[col].toFixed(2)) : row[col] ?? "—"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-800">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page <= 1}
                  className="btn-secondary !py-1.5 !px-3 text-xs disabled:opacity-30"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage(page + 1)}
                  disabled={tableData.data.length < tableData.per_page}
                  className="btn-secondary !py-1.5 !px-3 text-xs disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            </div>
            </div>
          )}
        </>
      )}
    </DashboardLayout>
  );
}
