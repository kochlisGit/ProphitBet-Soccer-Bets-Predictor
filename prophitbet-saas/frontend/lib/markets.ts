export const MARKET_OPTIONS = [
  ["result", "Match result (1X2)"], ["over-under", "Total goals 2.5"],
  ["goals-1.5", "Total goals 1.5"], ["goals-3.5", "Total goals 3.5"], ["goals-4.5", "Total goals 4.5"],
  ["btts", "Both teams to score"], ["home-goals-0.5", "Home goals 0.5"], ["home-goals-1.5", "Home goals 1.5"],
  ["away-goals-0.5", "Away goals 0.5"], ["away-goals-1.5", "Away goals 1.5"],
  ["corners-8.5", "Total corners 8.5"], ["corners-9.5", "Total corners 9.5"],
  ["corners-10.5", "Total corners 10.5"], ["corners-11.5", "Total corners 11.5"],
  ["shots-target-7.5", "Total shots on target 7.5"], ["shots-target-8.5", "Total shots on target 8.5"],
] as const;
export const marketName = (market = "result") => MARKET_OPTIONS.find(([id]) => id === (market === "over_under" ? "over-under" : market))?.[1] || market;
export const marketPick = (market = "result", pick: string) => market === "result"
  ? ({H: "Home win", D: "Draw", A: "Away win"}[pick] || pick)
  : `${marketName(market)} · ${pick}`;
export function marketSelection(market: string, pick: string) {
  if (market === "result") return pick;
  if (market === "btts") return `BTTS ${pick}`;
  if (market === "over-under") return `${pick} 2.5`;
  const parts = market.split("-"); const line = parts.pop();
  const prefix = ({goals: "", corners: "Corners ", "home-goals": "Home goals ", "away-goals": "Away goals ", "shots-target": "Shots on target "} as Record<string, string>)[parts.join("-")];
  return `${prefix}${pick} ${line}`;
}
export const MODEL_SELECTIONS = MARKET_OPTIONS.flatMap(([market]) =>
  (market === "result" ? ["H", "D", "A"] : market === "btts" ? ["No", "Yes"] : ["Under", "Over"])
    .map(pick => marketSelection(market, pick)));
