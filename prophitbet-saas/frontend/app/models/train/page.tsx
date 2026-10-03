"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { leaguesApi, modelsApi } from "@/lib/api";
import TaskProgress, { TaskProgressData } from "@/components/TaskProgress";

const MODEL_TYPES = [
  { value: "logistic", label: "Logistic Regression", tier: "pro" },
  { value: "decision_tree", label: "Decision Tree", tier: "pro" },
  { value: "naive_bayes", label: "Naive Bayes", tier: "pro" },
  { value: "knn", label: "K-Nearest Neighbors", tier: "pro" },
  { value: "svm", label: "Support Vector Machine", tier: "elite" },
  { value: "random_forest", label: "Random Forest", tier: "elite" },
  { value: "xgboost", label: "XGBoost", tier: "elite" },
  { value: "discriminant", label: "Discriminant Analysis", tier: "elite" },
];

export default function TrainModelPage() {
  const { token, user } = useAuth();
  const router = useRouter();
  const [leagues, setLeagues] = useState<any[]>([]);
  const [leagueId, setLeagueId] = useState("");
  const [modelType, setModelType] = useState("logistic");
  const [targetType, setTargetType] = useState("result");
  const [autoTune, setAutoTune] = useState(false);
  const [nTrials, setNTrials] = useState(50);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const plan = user?.plan || "free";

  useEffect(() => {
    if (!token) return;
    leaguesApi.list(token).then(setLeagues);
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token || !leagueId) return;
    setLoading(true);
    setStatus(null);

    try {
      let result;
      if (autoTune) {
        result = await modelsApi.autoTune(
          { league_id: Number(leagueId), model_type: modelType, target_type: targetType, n_trials: nTrials },
          token
        );
      } else {
        result = await modelsApi.train(
          { league_id: Number(leagueId), model_type: modelType, target_type: targetType },
          token
        );
      }
      setJobId(result.job_id);
      setStatus(null);
    } catch (err: any) {
      setStatus({ type: "error", message: err.message || "Training failed" });
    } finally {
      setLoading(false);
    }
  };

  const trainingFinished = (progress: TaskProgressData) => {
    setJobId(null);
    if (progress.status === "SUCCESS") {
      setStatus({ type: "success", message: "Training completed successfully. The new model is available on the Models page." });
    } else {
      setStatus({ type: "error", message: progress.error || "Training failed" });
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-xl">
        <h1 className="anim-page-title text-2xl font-bold text-white mb-2">Train a Model</h1>
        <p className="anim-page-sub text-gray-500 mb-8">
          Select a league, algorithm, and target — then let the machine learn.
        </p>

        {status && (
          <div
            className={`mb-6 p-4 rounded-xl text-sm ${
              status.type === "success"
                ? "bg-green-500/10 border border-green-500/20 text-green-400"
                : "bg-red-500/10 border border-red-500/20 text-red-400"
            }`}
          >
            {status.message}
          </div>
        )}

        {jobId && token && (
          <div className="mb-6">
            <TaskProgress
              jobId={jobId}
              token={token}
              label={autoTune ? "Auto-Tune & Train" : "Train Model"}
              statusPath={`/models/${jobId}/status`}
              onFinished={trainingFinished}
            />
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="anim-reveal-section">
            <div className="anim-reveal-inner card space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">League</label>
              <select className="input" value={leagueId} onChange={(e) => setLeagueId(e.target.value)} required>
                <option value="">Select a league</option>
                {leagues.map((l: any) => (
                  <option key={l.id} value={l.id}>
                    {l.country} — {l.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Algorithm</label>
              <select className="input" value={modelType} onChange={(e) => setModelType(e.target.value)}>
                {MODEL_TYPES.map((mt) => (
                  <option
                    key={mt.value}
                    value={mt.value}
                    disabled={mt.tier === "elite" && plan !== "elite"}
                  >
                    {mt.label} {mt.tier === "elite" && plan !== "elite" ? "(Elite)" : ""}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Target</label>
              <select className="input" value={targetType} onChange={(e) => setTargetType(e.target.value)}>
                <option value="result">Match Result (H/D/A)</option>
                <option value="over-under">Over/Under 2.5 Goals</option>
                {[
                  ["goals-1.5", "Total goals 1.5"], ["goals-3.5", "Total goals 3.5"], ["goals-4.5", "Total goals 4.5"],
                  ["btts", "Both teams to score (Yes/No)"],
                  ["home-goals-0.5", "Home goals 0.5"], ["home-goals-1.5", "Home goals 1.5"],
                  ["away-goals-0.5", "Away goals 0.5"], ["away-goals-1.5", "Away goals 1.5"],
                  ["corners-8.5", "Total corners 8.5"], ["corners-9.5", "Total corners 9.5"],
                  ["corners-10.5", "Total corners 10.5"], ["corners-11.5", "Total corners 11.5"],
                  ["shots-target-7.5", "Total shots on target 7.5"], ["shots-target-8.5", "Total shots on target 8.5"],
                ].map(([value, label]) => <option key={value} value={value}>{label}{value === "btts" ? "" : " · Over/Under"}</option>)}
              </select>
              <p className="mt-2 text-xs text-gray-500">Regulation time only. Training requires recorded outcomes for this market and at least 50 complete rows. Corners and shots coverage varies by league. New models need training; this does not add bookmaker odds.</p>
            </div>

            <div className="flex items-center gap-3">
              <input
                id="autotune"
                type="checkbox"
                checked={autoTune}
                onChange={(e) => setAutoTune(e.target.checked)}
                className="w-4 h-4 rounded border-gray-700 bg-gray-800 text-brand-500 focus:ring-brand-500"
              />
              <label htmlFor="autotune" className="text-sm text-gray-300">
                Auto-Tune with Optuna
              </label>
            </div>

            {autoTune && (
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-1">
                  Number of trials
                </label>
                <input
                  type="number"
                  className="input !w-32"
                  value={nTrials}
                  onChange={(e) => setNTrials(Number(e.target.value))}
                  min={10}
                  max={200}
                />
              </div>
            )}
          </div>
          </div>

          <button
            type="submit"
            disabled={loading || !!jobId || !leagueId}
            className="anim-page-actions btn-primary w-full disabled:opacity-50"
          >
            {loading ? "Starting training..." : autoTune ? "Auto-Tune & Train" : "Start Training"}
          </button>
        </form>
      </div>
    </DashboardLayout>
  );
}
