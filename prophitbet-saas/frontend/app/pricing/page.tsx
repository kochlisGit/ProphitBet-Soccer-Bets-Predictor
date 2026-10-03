"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Navbar from "@/components/Navbar";
import MotionPageRoot from "@/components/MotionPageRoot";
import { cn } from "@/lib/utils";

const tiers = [
  {
    name: "Free",
    price: "$0",
    period: "forever",
    cta: "Get started",
    href: "/register",
    highlight: false,
    features: [
      "Top 5 league predictions",
      "Result predictions only",
      "Descriptive statistics",
      "Last 7 days history",
    ],
    limits: [
      "No probabilities shown",
      "No custom model training",
      "No fixture predictions",
      "No API access",
    ],
  },
  {
    name: "Pro",
    price: "$9.99",
    period: "/month",
    cta: "Upgrade to Pro",
    href: "/register",
    highlight: true,
    features: [
      "All 36 league predictions",
      "Result + Over/Under",
      "Probability breakdowns",
      "All 8 analysis tools",
      "Train 3 models/month",
      "4 model types (LR, Tree, NB, KNN)",
      "Auto-Tune (Optuna)",
      "Fixture predictions",
      "Last 90 days history",
    ],
    limits: [],
  },
  {
    name: "Elite",
    price: "$24.99",
    period: "/month",
    cta: "Go Elite",
    href: "/register",
    highlight: false,
    features: [
      "Everything in Pro, plus:",
      "Unlimited model training",
      "All 8 model types (+ XGBoost, SVM, RF)",
      "Model explainability (SHAP)",
      "Full prediction history",
      "API access (1,000 req/day)",
      "Priority email + chat support",
    ],
    limits: [],
  },
];

export default function PricingPage() {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-zinc-900">
      <Navbar />
      <MotionPageRoot reviveKey={pathname} className="max-w-7xl mx-auto px-4 py-20">
        <h1 className="anim-page-title text-4xl font-extrabold text-center text-zinc-50 mb-4">
          Simple, transparent pricing
        </h1>
        <p className="anim-page-sub text-center text-zinc-400 mb-16 max-w-lg mx-auto">
          Start free. Upgrade when you need more power.
        </p>

        <div className="anim-features-section grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={cn(
                "anim-feature-card card relative flex flex-col",
                tier.highlight && "border-brand-500 ring-1 ring-brand-500/20"
              )}
            >
              {tier.highlight && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 bg-brand-600 rounded-full text-xs font-semibold text-white">
                  Most popular
                </div>
              )}
              <h3 className="text-xl font-bold text-zinc-50">{tier.name}</h3>
              <div className="mt-4 mb-6">
                <span className="text-4xl font-extrabold text-zinc-50">{tier.price}</span>
                <span className="text-zinc-400 ml-1">{tier.period}</span>
              </div>
              <Link
                href={tier.href}
                className={cn(
                  "w-full text-center py-2.5 rounded-lg font-semibold transition text-sm",
                  tier.highlight ? "btn-primary" : "btn-secondary"
                )}
              >
                {tier.cta}
              </Link>
              <ul className="mt-8 space-y-3 flex-1">
                {tier.features.map((f) => (
                  <li key={f} className="flex items-start gap-2 text-sm text-zinc-300">
                    <svg
                      className="w-4 h-4 text-brand-400 mt-0.5 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                    {f}
                  </li>
                ))}
                {tier.limits.map((l) => (
                  <li key={l} className="flex items-start gap-2 text-sm text-zinc-500">
                    <svg
                      className="w-4 h-4 text-zinc-600 mt-0.5 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    {l}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </MotionPageRoot>
    </div>
  );
}
