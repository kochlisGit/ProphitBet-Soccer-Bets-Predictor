"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { modelsApi, apiFetch } from "@/lib/api";
import { formatDate } from "@/lib/utils";

const PLOT_TYPES = [
  { value: "bar", label: "SHAP Feature Importance" },
  { value: "partial_dependence", label: "Partial Dependence" },
  { value: "boundary", label: "Decision Boundary" },
];

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { token, user } = useAuth();
  const router = useRouter();
  const [model, setModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [plotType, setPlotType] = useState("bar");
  const [plotImage, setPlotImage] = useState<string | null>(null);
  const [plotLoading, setPlotLoading] = useState(false);
  const [plotError, setPlotError] = useState("");

  const plan = user?.plan || "free";

  useEffect(() => {
    if (!token || !id) return;
    modelsApi
      .get(id, token)
      .then(setModel)
      .catch(() => setModel(null))
      .finally(() => setLoading(false));
  }, [token, id]);

  const generatePlot = async () => {
    if (!token || !id) return;
    setPlotLoading(true);
    setPlotError("");
    try {
      const res = await apiFetch(`/models/${id}/explain?plot_type=${plotType}`, { token });
      setPlotImage(res.image_base64);
    } catch (err: any) {
      setPlotError(err.message || "Failed to generate plot");
    } finally {
      setPlotLoading(false);
    }
  };

  const deleteModel = async () => {
    if (!token || !id || !confirm("Delete this model permanently?")) return;
    try {
      await modelsApi.delete(id, token);
      router.push("/models");
    } catch {}
  };

  return (
    <DashboardLayout>
      {loading ? (
        <div className="text-center py-20 text-zinc-400">Loading...</div>
      ) : !model ? (
        <div className="text-center py-20 text-zinc-400">Model not found</div>
      ) : (
        <div className="max-w-3xl">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h1 className="anim-page-title text-2xl font-bold text-zinc-50 capitalize">
                {model.model_type.replace(/_/g, " ")}
              </h1>
              <p className="anim-page-sub text-zinc-400 mt-1">
                League #{model.league_id} &middot; {model.target_type} &middot;
                Created {formatDate(model.created_at)}
              </p>
            </div>
            <button
              onClick={deleteModel}
              className="anim-page-actions text-sm text-zinc-500 hover:text-red-400 transition"
            >
              Delete model
            </button>
          </div>

          {/* Metrics */}
          {model.metrics && Object.keys(model.metrics).length > 0 && (
            <div className="anim-reveal-section mb-6">
              <div className="anim-reveal-inner card">
              <h2 className="text-lg font-semibold text-zinc-50 mb-4">Performance Metrics</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(model.metrics).map(([key, val]) => (
                  <div key={key} className="bg-zinc-700 rounded-lg px-4 py-3">
                    <p className="text-xs text-zinc-400 uppercase mb-1">{key}</p>
                    <p className="text-xl font-bold text-zinc-50">{(val as number).toFixed(4)}</p>
                  </div>
                ))}
              </div>
              </div>
            </div>
          )}

          {/* Hyperparameters */}
          {model.hyperparams && Object.keys(model.hyperparams).length > 0 && (
            <div className="anim-reveal-section mb-6">
              <div className="anim-reveal-inner card">
              <h2 className="text-lg font-semibold text-zinc-50 mb-4">Hyperparameters</h2>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(model.hyperparams).map(([key, val]) => (
                  <div key={key} className="flex justify-between bg-zinc-700 rounded px-3 py-2">
                    <span className="text-sm text-zinc-400">{key}</span>
                    <span className="text-sm text-zinc-100 font-mono">{String(val)}</span>
                  </div>
                ))}
              </div>
              </div>
            </div>
          )}

          {/* Explainability (Elite only) */}
          <div className="anim-reveal-section">
            <div className="anim-reveal-inner card">
            <h2 className="text-lg font-semibold text-zinc-50 mb-4">Model Explainability</h2>
            {plan !== "elite" ? (
              <div className="text-center py-8">
                <p className="text-zinc-400 mb-3">
                  SHAP-based model explainability is available on the Elite plan.
                </p>
                <a href="/billing" className="btn-primary text-sm">Upgrade to Elite</a>
              </div>
            ) : (
              <>
                <div className="flex items-end gap-3 mb-4">
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-zinc-400 mb-1">Plot Type</label>
                    <select
                      className="input"
                      value={plotType}
                      onChange={(e) => setPlotType(e.target.value)}
                    >
                      {PLOT_TYPES.map((pt) => (
                        <option key={pt.value} value={pt.value}>{pt.label}</option>
                      ))}
                    </select>
                  </div>
                  <button
                    onClick={generatePlot}
                    disabled={plotLoading}
                    className="btn-primary disabled:opacity-50"
                  >
                    {plotLoading ? "Generating..." : "Generate Plot"}
                  </button>
                </div>

                {plotError && (
                  <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400 mb-4">
                    {plotError}
                  </div>
                )}

                {plotImage && (
                  <div className="bg-gray-800 rounded-lg p-4 flex items-center justify-center">
                    <img
                      src={`data:image/png;base64,${plotImage}`}
                      alt="Model explainability plot"
                      className="max-w-full rounded"
                    />
                  </div>
                )}
              </>
            )}
            </div>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
