"use client";

import { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { authApi } from "@/lib/api";

export default function SettingsPage() {
  const { user, token, refreshUser } = useAuth();
  const [name, setName] = useState("");
  const [notifyDaily, setNotifyDaily] = useState(true);
  const [notifyTraining, setNotifyTraining] = useState(true);
  const [notifyWeekly, setNotifyWeekly] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!user) return;
    setName(user.name || "");
    const p = user.preferences;
    if (p) {
      setNotifyDaily(!!p.notify_daily_digest);
      setNotifyTraining(!!p.notify_training);
      setNotifyWeekly(!!p.notify_weekly_report);
    }
  }, [user]);

  const handleSave = async () => {
    if (!token) return;
    setSaving(true);
    setError("");
    try {
      await authApi.updateProfile(
        {
          name,
          notify_daily_digest: notifyDaily,
          notify_training: notifyTraining,
          notify_weekly_report: notifyWeekly,
        },
        token
      );
      await refreshUser();
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e: any) {
      setError(e.message || "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-xl">
        <h1 className="anim-page-title text-2xl font-bold text-zinc-50 mb-2">Settings</h1>
        <p className="anim-page-sub text-zinc-400 mb-8">Manage your profile and preferences</p>

        {/* Profile */}
        <div className="anim-reveal-section mb-6">
          <div className="anim-reveal-inner card">
          <h2 className="text-lg font-semibold text-zinc-50 mb-4">Profile</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-1">Name</label>
              <input
                type="text"
                className="input"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-1">Email</label>
              <input type="email" className="input" defaultValue={user?.email || ""} disabled />
              <p className="text-xs text-zinc-500 mt-1">Email cannot be changed</p>
            </div>
            {error && <p className="text-sm text-red-400">{error}</p>}
            <button
              className="btn-primary text-sm"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? "Saving..." : saved ? "Saved!" : "Save Changes"}
            </button>
          </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="anim-reveal-section mb-6">
          <div className="anim-reveal-inner card">
          <h2 className="text-lg font-semibold text-zinc-50 mb-4">Notifications</h2>
          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifyDaily}
                onChange={(e) => setNotifyDaily(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-600 bg-zinc-700 text-brand-500"
              />
              <span className="text-sm text-zinc-300">Daily prediction digest email</span>
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifyTraining}
                onChange={(e) => setNotifyTraining(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-600 bg-zinc-700 text-brand-500"
              />
              <span className="text-sm text-zinc-300">Model training completion alerts</span>
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={notifyWeekly}
                onChange={(e) => setNotifyWeekly(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-600 bg-zinc-700 text-brand-500"
              />
              <span className="text-sm text-zinc-300">Weekly accuracy report</span>
            </label>
          </div>
          </div>
        </div>

        {/* API Keys (Elite only) */}
        {user?.plan === "elite" && (
          <div className="anim-reveal-section">
            <div className="anim-reveal-inner card">
            <h2 className="text-lg font-semibold text-zinc-50 mb-4">API Access</h2>
            <p className="text-sm text-zinc-400 mb-3">
              Use the API to integrate predictions into your own applications.
            </p>
            <div className="bg-zinc-700 rounded-lg p-3 font-mono text-sm text-zinc-300 break-all">
              Use your Bearer token from /auth/login for API access. Rate limit: 1,000 requests/day.
            </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
