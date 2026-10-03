"use client";

import { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { billingApi } from "@/lib/api";
import { cn, planBadgeColor } from "@/lib/utils";

export default function BillingPage() {
  const { token, user } = useAuth();
  const [usage, setUsage] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const plan = user?.plan || "free";

  useEffect(() => {
    if (!token) return;
    billingApi.usage(token).then(setUsage).catch(() => {});
  }, [token]);

  const handleUpgrade = async (targetPlan: string) => {
    if (!token) return;
    setLoading(true);
    try {
      const res = await billingApi.checkout(targetPlan, token);
      window.location.href = res.checkout_url;
    } catch {
      setLoading(false);
    }
  };

  const handleManage = async () => {
    if (!token) return;
    try {
      const res = await billingApi.portal(token);
      window.location.href = res.url;
    } catch {}
  };

  return (
    <DashboardLayout>
      <div className="max-w-2xl">
        <h1 className="anim-page-title text-2xl font-bold text-white mb-2">Billing</h1>
        <p className="anim-page-sub text-gray-500 mb-8">Manage your subscription and view usage</p>

        {/* Current Plan */}
        <div className="anim-reveal-section mb-6">
          <div className="anim-reveal-inner card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Current Plan</p>
              <p className="text-2xl font-bold text-white mt-1 capitalize">{plan}</p>
            </div>
            <span className={`badge text-sm ${planBadgeColor(plan)}`}>
              {plan === "free" ? "FREE" : "ACTIVE"}
            </span>
          </div>
          </div>
        </div>

        {/* Usage */}
        {usage && (
          <div className="anim-reveal-section mb-6">
            <div className="anim-reveal-inner card">
            <h2 className="text-lg font-semibold text-white mb-4">Usage This Month</h2>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Models Trained</span>
                  <span className="text-white">
                    {usage.models_trained_this_month} / {usage.models_limit === 999999 ? "Unlimited" : usage.models_limit}
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-brand-500 h-2 rounded-full"
                    style={{
                      width: `${Math.min(100, (usage.models_trained_this_month / Math.max(1, usage.models_limit)) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            </div>
            </div>
          </div>
        )}

        {/* Upgrade / Manage */}
        <div className="anim-page-actions space-y-3">
          {plan === "free" && (
            <>
              <button
                onClick={() => handleUpgrade("pro")}
                disabled={loading}
                className="btn-primary w-full disabled:opacity-50"
              >
                Upgrade to Pro — $9.99/mo
              </button>
              <button
                onClick={() => handleUpgrade("elite")}
                disabled={loading}
                className="btn-secondary w-full disabled:opacity-50"
              >
                Upgrade to Elite — $24.99/mo
              </button>
            </>
          )}
          {plan === "pro" && (
            <>
              <button
                onClick={() => handleUpgrade("elite")}
                disabled={loading}
                className="btn-primary w-full disabled:opacity-50"
              >
                Upgrade to Elite — $24.99/mo
              </button>
              <button onClick={handleManage} className="btn-secondary w-full">
                Manage Subscription
              </button>
            </>
          )}
          {plan === "elite" && (
            <button onClick={handleManage} className="btn-secondary w-full">
              Manage Subscription
            </button>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
