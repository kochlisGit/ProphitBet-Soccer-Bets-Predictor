"use client";

import Link from "next/link";
import Navbar from "@/components/Navbar";

const features = [
  {
    title: "Daily forecasts",
    desc: "Models trained on league stats refresh with the schedule—covering 36+ competitions in one place.",
    icon: "M13 10V3L4 14h7v7l9-11h-7z",
  },
  {
    title: "Stat toolkit",
    desc: "Correlation, distributions, variance, feature selection, and more—interactive charts, not static PDFs.",
    icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
  },
  {
    title: "Train your own",
    desc: "Pick algorithms from logistic regression to tree ensembles and tune on your chosen leagues.",
    icon: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z",
  },
  {
    title: "Auto-tune",
    desc: "Optuna-driven search finds stronger hyperparameters without a spreadsheet full of manual runs.",
    icon: "M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4",
  },
  {
    title: "Fixtures & odds",
    desc: "Upcoming fixtures stay in sync with odds and pre-match outputs so you are not copy-pasting from five tabs.",
    icon: "M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z",
  },
  {
    title: "Explainability",
    desc: "SHAP, partial dependence, and boundary views when you need to see what drove a pick—not just the headline.",
    icon: "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z",
  },
];

export default function LandingPageView() {
  return (
    <div className="min-h-screen">
      <Navbar />

      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-emerald-950/35 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-7xl px-4 pb-16 pt-24 text-center sm:pb-24 sm:pt-32">
          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-zinc-500">
            Match intelligence
          </p>
          <h1 className="text-5xl font-bold tracking-tight sm:text-6xl md:text-7xl">
            <span className="anim-hero-line inline-block text-zinc-50">Forecast with</span>{" "}
            <span className="anim-hero-line inline-block text-emerald-400">clarity</span>
          </h1>
          <p className="anim-hero-sub mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-zinc-400">
            One workspace for league data, fixtures, and models—so you spend time on edges, not on glue code.
          </p>
          <div className="anim-hero-actions mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link href="/register" className="btn-primary px-8 py-3 text-base">
              Start for free
            </Link>
            <Link href="/pricing" className="btn-secondary px-8 py-3 text-base">
              Compare plans
            </Link>
          </div>
          <p className="anim-hero-note mt-4 text-sm text-zinc-600">
            Free tier, no card. Upgrade when you outgrow it.
          </p>
        </div>
      </section>

      <section className="anim-features-section mx-auto max-w-7xl px-4 py-20">
        <h2 className="mb-3 text-center text-3xl font-bold tracking-tight text-zinc-50">
          Built for the full workflow
        </h2>
        <p className="mx-auto mb-16 max-w-xl text-center text-zinc-400">
          From ingestion to evaluation—so you are not duct-taping spreadsheets to a black box.
        </p>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="anim-feature-card card group transition-colors duration-300 hover:border-emerald-900/40"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-md bg-emerald-500/10 ring-1 ring-emerald-500/15 transition-colors group-hover:bg-emerald-500/15">
                <svg
                  className="h-5 w-5 text-emerald-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d={f.icon} />
                </svg>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-zinc-50">{f.title}</h3>
              <p className="text-sm leading-relaxed text-zinc-300">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="anim-cta-section mx-auto max-w-4xl px-4 py-20 text-center">
        <div className="anim-cta-inner card border-emerald-900/35 bg-gradient-to-br from-emerald-950/40 via-zinc-900/90 to-zinc-900 shadow-lg shadow-emerald-950/20">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-zinc-50">See it on your leagues</h2>
          <p className="mx-auto mb-8 max-w-lg text-zinc-300">
            Create an account, pick your competitions, and follow predictions alongside the stats that feed them.
          </p>
          <Link href="/register" className="btn-primary inline-block px-10 py-3 text-base">
            Create your account
          </Link>
        </div>
      </section>

      <footer className="border-t border-zinc-700/90 py-8">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 sm:flex-row">
          <p className="text-center text-sm text-zinc-500 sm:text-left">
            ProphitBet &copy; {new Date().getFullYear()}. For entertainment only—not betting or financial advice.
          </p>
          <div className="flex gap-6 text-sm text-zinc-500">
            <Link href="/pricing" className="transition-colors hover:text-zinc-300">
              Pricing
            </Link>
            <a href="#" className="transition-colors hover:text-zinc-300">
              Terms
            </a>
            <a href="#" className="transition-colors hover:text-zinc-300">
              Privacy
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
