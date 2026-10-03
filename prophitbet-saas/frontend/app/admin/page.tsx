"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { apiFetch } from "@/lib/api";
import TaskProgress, { TaskProgressData } from "@/components/TaskProgress";

type ActiveJob = { id: string; label: string };

export default function AdminPage() {
  const { token, user } = useAuth();
  const [stats, setStats] = useState<any>(null);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState("");
  const [activeJob, setActiveJob] = useState<ActiveJob | null>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem("prophitbet-admin-active-job");
    if (saved) {
      try { setActiveJob(JSON.parse(saved)); } catch { window.localStorage.removeItem("prophitbet-admin-active-job"); }
    }
  }, []);

  const trackJob = (id: string, label: string) => {
    const job = { id, label };
    setActiveJob(job);
    window.localStorage.setItem("prophitbet-admin-active-job", JSON.stringify(job));
    setActionMsg("");
  };

  const finishJob = (progress: TaskProgressData) => {
    window.localStorage.removeItem("prophitbet-admin-active-job");
    const result = progress.result || {};
    const details = Object.entries(result).map(([key, value]) => `${key.replaceAll("_", " ")}: ${Array.isArray(value) ? (value.join(", ") || "none") : typeof value === "object" ? JSON.stringify(value) : String(value)}`).join(" · ");
    setActionMsg(progress.status === "SUCCESS" ? `${activeJob?.label || "Task"} completed. ${details}` : `${activeJob?.label || "Task"} failed: ${progress.error || "Unknown error"}`);
    setActiveJob(null);
  };

  useEffect(() => {
    if (!token || !user?.is_admin) return;
    Promise.all([
      apiFetch("/admin/stats", { token }),
      apiFetch("/admin/users", { token }),
    ])
      .then(([s, u]) => {
        setStats(s);
        setUsers(u);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token, user]);

  const triggerSync = async () => {
    if (!token) return;
    setActionMsg("Starting league sync...");
    try {
      const res = await apiFetch("/admin/sync-leagues", { method: "POST", token });
      trackJob(res.job_id, "Sync League Data");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  const triggerPredictions = async () => {
    if (!token) return;
    setActionMsg("Starting prediction generation...");
    try {
      const res = await apiFetch("/admin/generate-predictions", { method: "POST", token });
      trackJob(res.job_id, "Generate Predictions");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  const triggerHouseModels = async () => {
    if (!token) return;
    setActionMsg("Starting house model training (this may take a while)...");
    try {
      const res = await apiFetch("/admin/train-house-models", { method: "POST", token });
      trackJob(res.job_id, "Train House Models");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  const triggerFixtures = async () => {
    if (!token) return;
    setActionMsg("Starting fixture scraping...");
    try {
      const res = await apiFetch("/admin/scrape-fixtures", { method: "POST", token });
      trackJob(res.job_id, "Scrape Fixtures");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  const triggerResults = async () => {
    if (!token) return;
    setActionMsg("Starting match result update...");
    try {
      const res = await apiFetch("/admin/update-results", { method: "POST", token });
      trackJob(res.job_id, "Update Results");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  const triggerMiroFishBatch = async () => {
    if (!token) return;
    setActionMsg("Starting MiroFish multi-agent batch simulations for upcoming matches...");
    try {
      const res = await apiFetch("/admin/mirofish/batch-simulate?limit=25", { method: "POST", token });
      trackJob(res.job_id, "Batch MiroFish Swarm");
    } catch (e: any) {
      setActionMsg(`Error: ${e.message}`);
    }
  };

  // Value alerts state
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loadingAlerts, setLoadingAlerts] = useState(false);
  const [webhookUrl, setWebhookUrl] = useState("");
  const [dispatchMsg, setDispatchMsg] = useState<string | null>(null);

  const fetchAlerts = async () => {
    if (!token) return;
    try {
      setLoadingAlerts(true);
      const res = await apiFetch("/admin/alerts/preview?min_ev=4.0", { token });
      setAlerts(res.alerts || []);
    } catch (e: any) {
      console.error("Failed to load alerts preview", e);
    } finally {
      setLoadingAlerts(false);
    }
  };

  const handleDispatch = async () => {
    if (!token || !webhookUrl) return;
    try {
      setDispatchMsg("Dispatching live webhook...");
      const res = await apiFetch("/admin/alerts/dispatch", {
        method: "POST",
        body: JSON.stringify({ webhook_url: webhookUrl, min_ev_pct: 4.0, platform: "discord" }),
        token,
      });
      setDispatchMsg(res.status === "dispatched" ? "Alert successfully dispatched to Discord!" : `Result: ${res.message || res.status}`);
      setTimeout(() => setDispatchMsg(null), 4000);
    } catch (e: any) {
      setDispatchMsg(`Dispatch failed: ${e.message}`);
      setTimeout(() => setDispatchMsg(null), 4000);
    }
  };


  if (!user?.is_admin) {
    return (
      <DashboardLayout>
        <div className="text-center py-20 text-gray-500">Admin access required.</div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="mb-8">
        <h1 className="anim-page-title text-2xl font-bold text-white">Admin Panel</h1>
        <p className="anim-page-sub text-gray-500 mt-1">Platform management and oversight</p>
      </div>

      {loading ? (
        <div className="text-center py-12 text-gray-500">Loading...</div>
      ) : (
        <>
          {/* Stats */}
          {stats && (
            <div className="anim-features-section grid sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Users</p>
                <p className="text-2xl font-bold text-white">{stats.total_users}</p>
              </div>
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Pro Users</p>
                <p className="text-2xl font-bold text-blue-400">{stats.pro_users}</p>
              </div>
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Elite Users</p>
                <p className="text-2xl font-bold text-purple-400">{stats.elite_users}</p>
              </div>
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Models</p>
                <p className="text-2xl font-bold text-white">{stats.total_models}</p>
              </div>
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Predictions</p>
                <p className="text-2xl font-bold text-white">{stats.total_predictions}</p>
              </div>
              <div className="anim-feature-card card">
                <p className="text-xs text-gray-500">Leagues</p>
                <p className="text-2xl font-bold text-white">{stats.total_leagues}</p>
              </div>
            </div>
          )}

          <div className="anim-reveal-section mb-6">
            <div className="anim-reveal-inner card border border-amber-900/40 bg-amber-950/10">
            <h2 className="text-lg font-semibold text-white mb-2">Prediction pipeline</h2>
            <p className="text-sm text-gray-400 mb-3">
              If Celery is still downloading, let <strong className="text-gray-300">Sync League Data</strong> finish first
              (often 15–30+ minutes for all leagues). Predictions stay empty until you run the steps below in order.
            </p>
            <ol className="list-decimal list-inside text-sm text-gray-400 space-y-2">
              <li>
                <span className="text-gray-200">Sync League Data</span> — CSVs to MinIO + datasets in the database.
              </li>
              <li>
                <span className="text-gray-200">Train House Models</span> — Separate Random Forest models for each supported market and league. Markets without recorded outcomes are skipped; this takes longer than 1X2-only training.
              </li>
              <li>
                <span className="text-gray-200">Scrape Fixtures</span> — Refresh provider schedules and retire superseded dates. The completion report lists leagues without available data.
              </li>
              <li>
                <span className="text-gray-200">Generate Predictions</span> — Runs inference and fills the predictions list.
              </li>
              <li>
                <span className="text-purple-300 font-medium">MiroFish Swarm Simulations</span> — Simulates multi-agent debates & computes hybrid ensemble predictions.
              </li>
            </ol>
            </div>
          </div>

          {/* Actions */}
          <div className="anim-reveal-section mb-6">
            <div className="anim-reveal-inner card">
            <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
            <p className="text-xs text-gray-500 mb-3">Use this order after sync completes:</p>
            <div className="flex flex-wrap gap-3">
              <button onClick={triggerSync} disabled={!!activeJob} className="btn-primary text-sm disabled:opacity-50">
                1 · Sync League Data
              </button>
              <button onClick={triggerHouseModels} disabled={!!activeJob} className="btn-secondary text-sm disabled:opacity-50">
                2 · Train House Models
              </button>
              <button onClick={triggerFixtures} disabled={!!activeJob} className="btn-secondary text-sm disabled:opacity-50">
                3 · Scrape Fixtures
              </button>
              <button onClick={triggerPredictions} disabled={!!activeJob} className="btn-secondary text-sm disabled:opacity-50">
                4 · Generate Predictions
              </button>
              <button onClick={triggerMiroFishBatch} disabled={!!activeJob} className="btn-secondary text-sm border-purple-500/40 text-purple-300 hover:bg-purple-900/20 disabled:opacity-50">
                5 · Batch MiroFish Swarm
              </button>
              <button onClick={triggerResults} disabled={!!activeJob} className="btn-secondary text-sm disabled:opacity-50">
                Update Results
              </button>
            </div>
            {actionMsg && (
              <p className="mt-3 text-sm text-gray-400">{actionMsg}</p>
            )}
            {activeJob && token && (
              <TaskProgress
                jobId={activeJob.id}
                token={token}
                label={activeJob.label}
                onFinished={finishJob}
              />
            )}
            </div>
          </div>

          {/* Telegram & Discord Value Alerts Dispatcher (Phase 6) */}
          <div className="anim-reveal-section mb-6">
            <div className="anim-reveal-inner card border border-purple-900/40 bg-zinc-900/90">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-3 mb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg">📢</span>
                    <h2 className="text-lg font-semibold text-white">Automated +EV Value Alerts Dispatcher</h2>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      Phase 6
                    </span>
                  </div>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Scans upcoming predictions for mathematical positive value (+EV ≥ 4%) and generates Discord webhook embeds and Telegram markdown cards.
                  </p>
                </div>
                <button
                  onClick={fetchAlerts}
                  disabled={loadingAlerts}
                  className="px-3.5 py-1.5 text-xs bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium transition-colors self-start sm:self-auto"
                >
                  {loadingAlerts ? "Scanning Fixtures..." : "🔍 Preview Value Alerts"}
                </button>
              </div>

              {/* Webhook Dispatch Bar */}
              <div className="flex flex-col sm:flex-row items-center gap-2 mb-4">
                <input
                  type="url"
                  placeholder="https://discord.com/api/webhooks/... (or Telegram webhook)"
                  value={webhookUrl}
                  onChange={(e) => setWebhookUrl(e.target.value)}
                  className="w-full sm:flex-1 px-3 py-2 text-xs bg-zinc-950 border border-zinc-800 rounded-lg text-white"
                />
                <button
                  onClick={handleDispatch}
                  disabled={!webhookUrl}
                  className="w-full sm:w-auto px-4 py-2 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-lg font-bold transition-colors"
                >
                  Dispatch Webhook
                </button>
              </div>

              {dispatchMsg && (
                <div className="p-3 mb-4 bg-purple-950/40 border border-purple-500/30 rounded-lg text-xs text-purple-300">
                  {dispatchMsg}
                </div>
              )}

              {/* Alerts List */}
              {alerts.length > 0 ? (
                <div className="space-y-3">
                  {alerts.map((item: any, idx: number) => {
                    const a = item.alert;
                    return (
                      <div key={idx} className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 text-xs space-y-2 font-mono">
                        <div className="flex items-center justify-between font-sans">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white text-sm">{a.match}</span>
                            <span className="text-zinc-500 text-[11px] font-mono">({a.league})</span>
                          </div>
                          <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                            +{a.ev_edge_pct}% EV
                          </span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-zinc-300 pt-1">
                          <div>Pick: <strong className="text-amber-300">{a.pick_name} ({a.pick})</strong></div>
                          <div>Market Odds: <strong className="text-white">@{a.market_odds}</strong></div>
                          <div>Quarter-Kelly: <strong className="text-purple-300">{a.quarter_kelly_stake_pct}% bankroll</strong></div>
                          <div>Kickoff: <span className="text-zinc-400">{a.match_date}</span></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-zinc-500 bg-zinc-950/30 rounded-xl border border-zinc-850">
                  Click "Preview Value Alerts" above to scan upcoming predictions for +EV betting opportunities.
                </div>
              )}
            </div>
          </div>

          {/* User List */}
          <div className="anim-reveal-section">
            <div className="anim-reveal-inner card">
            <h2 className="text-lg font-semibold text-white mb-4">Users</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b border-gray-800">
                    <th className="pb-3 font-medium">Email</th>
                    <th className="pb-3 font-medium">Name</th>
                    <th className="pb-3 font-medium">Plan</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium">Joined</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/50">
                  {users.map((u: any) => (
                    <tr key={u.id} className="hover:bg-gray-800/30">
                      <td className="py-3 text-white">{u.email}</td>
                      <td className="py-3 text-gray-400">{u.name || "—"}</td>
                      <td className="py-3">
                        <span className="badge bg-gray-800 border-gray-700 capitalize">{u.plan}</span>
                      </td>
                      <td className="py-3">
                        {u.is_active ? (
                          <span className="text-green-400 text-xs">Active</span>
                        ) : (
                          <span className="text-red-400 text-xs">Disabled</span>
                        )}
                      </td>
                      <td className="py-3 text-gray-500 text-xs">
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            </div>
          </div>
        </>
      )}
    </DashboardLayout>
  );
}
