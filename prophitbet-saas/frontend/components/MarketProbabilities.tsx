export default function MarketProbabilities({ probabilities }: { probabilities?: Record<string, number | undefined> | null }) {
  const entries = Object.entries(probabilities || {}).filter(([, value]) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1);
  if (!entries.length) return <span className="text-xs text-zinc-500">Probabilities unavailable</span>;
  return <div className="min-w-[140px] space-y-1 text-xs text-zinc-300">{entries.map(([label, value]) => <div key={label} className="flex justify-between gap-3"><span>{label}</span><span className="tabular-nums">{(value! * 100).toFixed(1)}%</span></div>)}</div>;
}
