"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { modelsApi } from "@/lib/api";
import { formatDate } from "@/lib/utils";

export default function ModelsPage() {
  const { token } = useAuth();
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) return;
    modelsApi
      .list(token)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [token]);

  const deleteModel = async (id: string) => {
    if (!token || !confirm("Delete this model?")) return;
    try {
      await modelsApi.delete(id, token);
      setModels(models.filter((m) => m.id !== id));
    } catch {}
  };

  return (
    <DashboardLayout>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="anim-page-title text-2xl font-bold text-zinc-50">Your Models</h1>
          <p className="anim-page-sub text-zinc-400 mt-1">{models.length} trained models</p>
        </div>
        <Link href="/models/train" className="anim-page-actions btn-primary text-sm">
          Train New Model
        </Link>
      </div>

      {loading ? (
        <div className="text-center py-12 text-zinc-400">Loading models...</div>
      ) : models.length === 0 ? (
        <div className="anim-reveal-section">
          <div className="anim-reveal-inner card text-center py-12">
          <p className="text-zinc-400 mb-4">You haven&apos;t trained any models yet.</p>
          <Link href="/models/train" className="btn-primary text-sm">
            Train your first model
          </Link>
          </div>
        </div>
      ) : (
        <div className="anim-features-section grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m: any) => (
            <div key={m.id} className="anim-feature-card card">
              <div className="flex items-start justify-between mb-3">
                <Link href={`/models/${m.id}`}>
                  <h3 className="text-white font-semibold capitalize hover:text-brand-400 transition cursor-pointer">
                    {m.model_type.replace(/_/g, " ")}
                  </h3>
                  <p className="text-xs text-gray-500 mt-0.5">
                    League #{m.league_id} &middot; {m.target_type}
                  </p>
                </Link>
                {!m.is_house_model && (
                  <button
                    onClick={() => deleteModel(m.id)}
                    className="text-zinc-500 hover:text-red-400 transition text-sm"
                  >
                    Delete
                  </button>
                )}
              </div>
              {m.metrics && (
                <div className="grid grid-cols-2 gap-2 mb-3">
                  {Object.entries(m.metrics).slice(0, 4).map(([key, val]) => (
                    <div key={key} className="bg-zinc-700 rounded px-2.5 py-1.5">
                      <p className="text-[10px] text-zinc-400 uppercase">{key}</p>
                      <p className="text-sm font-semibold text-zinc-50">{(val as number).toFixed(3)}</p>
                    </div>
                  ))}
                </div>
              )}
              <p className="text-xs text-zinc-500">Created {formatDate(m.created_at)}</p>
            </div>
          ))}
        </div>
      )}
    </DashboardLayout>
  );
}
