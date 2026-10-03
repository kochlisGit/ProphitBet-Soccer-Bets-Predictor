import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function formatUtcToLocal(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}


export function resultColor(result: string): string {
  switch (result) {
    case "H": return "text-green-400";
    case "A": return "text-red-400";
    case "D": return "text-yellow-400";
    case "Over": return "text-green-400";
    case "Under": return "text-red-400";
    default: return "text-gray-400";
  }
}

export function planBadgeColor(plan: string): string {
  switch (plan) {
    case "elite": return "bg-purple-500/20 text-purple-300 border-purple-500/30";
    case "pro": return "bg-blue-500/20 text-blue-300 border-blue-500/30";
    default: return "bg-gray-500/20 text-gray-300 border-gray-500/30";
  }
}

export interface PopularLeagueConfig {
  name: string;
  country: string;
  flag: string;
  rank: number;
  displayName: string;
}

export const TOP_LEAGUES: PopularLeagueConfig[] = [
  { name: "Betway-Premiership", country: "South Africa", flag: "🇿🇦", rank: 1, displayName: "Betway Premiership (PSL)" },
  { name: "Premier-League", country: "England", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", rank: 2, displayName: "Premier League" },
  { name: "La-Liga", country: "Spain", flag: "🇪🇸", rank: 3, displayName: "La Liga" },
  { name: "Serie-A", country: "Italy", flag: "🇮🇹", rank: 4, displayName: "Serie A" },
  { name: "Bundesliga-1", country: "Germany", flag: "🇩🇪", rank: 5, displayName: "Bundesliga" },
  { name: "Ligue-1", country: "France", flag: "🇫🇷", rank: 6, displayName: "Ligue 1" },
  { name: "Championship", country: "England", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", rank: 7, displayName: "Championship" },
  { name: "Eredivisie", country: "Netherlands", flag: "🇳🇱", rank: 8, displayName: "Eredivisie" },
  { name: "Liga-1", country: "Portugal", flag: "🇵🇹", rank: 9, displayName: "Liga Portugal" },
];

export function cleanLeagueName(name: string, country?: string): string {
  if (!name) return "";
  const cleaned = name.replace(/-/g, " ").trim();
  const c = (country || "").toLowerCase();

  if (cleaned.toLowerCase() === "bundesliga 1") return "Bundesliga";
  if (cleaned.toLowerCase() === "betway premiership") return "Betway Premiership";

  // Disambiguate duplicate league names by country
  if (cleaned.toLowerCase() === "premier league" && (c === "russia" || c.includes("russ"))) {
    return "Russian Premier League";
  }
  if (cleaned.toLowerCase() === "serie a" && (c === "brazil" || c.includes("braz"))) {
    return "Serie A (Brazil)";
  }
  if (cleaned.toLowerCase() === "liga 1") {
    if (c === "portugal" || c.includes("port")) return "Liga Portugal";
    if (c === "romania" || c.includes("roman")) return "Liga 1 (Romania)";
  }
  if (cleaned.toLowerCase() === "super league") {
    if (c === "greece" || c.includes("gree")) return "Super League (Greece)";
    if (c === "switzerland" || c.includes("switz")) return "Swiss Super League";
    if (c === "china" || c.includes("chin")) return "Chinese Super League";
  }
  if (cleaned.toLowerCase() === "premiership" && (c === "scotland" || c.includes("scot"))) {
    return "Scottish Premiership";
  }

  return cleaned;
}

export function getLeaguePriority(leagueName: string, country?: string): number {
  const normName = leagueName.toLowerCase().replace(/[\s-_]/g, "");
  const normCountry = (country || "").toLowerCase();

  // Explicit priority checks
  if (normName.includes("betway") || normName.includes("psl") || normCountry.includes("south africa")) return 1;
  if (normName.includes("premierleague") && (normCountry.includes("england") || !normCountry)) return 2;
  if (normName.includes("laliga") || (normName.includes("primera") && normCountry.includes("spain"))) return 3;
  if (normName.includes("seriea") && (normCountry.includes("italy") || !normCountry)) return 4;
  if (normName.includes("bundesliga1") || (normName === "bundesliga" && (normCountry.includes("germany") || !normCountry))) return 5;
  if (normName.includes("ligue1") && (normCountry.includes("france") || !normCountry)) return 6;
  if (normName.includes("championship") && (normCountry.includes("england") || !normCountry)) return 7;
  if (normName.includes("eredivisie")) return 8;
  if (normName.includes("liga1") && normCountry.includes("portugal")) return 9;
  if (normName.includes("bundesliga2")) return 10;
  if (normName.includes("segundadivision")) return 11;
  if (normName.includes("serieb")) return 12;
  if (normName.includes("ligue2")) return 13;
  if (normName.includes("league1") && normCountry.includes("england")) return 14;
  if (normName.includes("league2") && normCountry.includes("england")) return 15;

  return 100;
}

export function getCountryFlag(countryOrLeague: string): string {
  if (!countryOrLeague) return "⚽";
  const str = countryOrLeague.toLowerCase();

  // Exact or contains country names
  if (str === "south africa" || str.includes("south africa") || str.includes("betway") || str.includes("psl")) return "🇿🇦";
  if (str === "england" || str.includes("premier-league") || str === "premier league" || str.includes("championship")) return "🏴󠁧󠁢󠁥󠁮󠁧󠁿";
  if (str === "spain" || str.includes("la-liga") || str.includes("la liga") || str.includes("segunda")) return "🇪🇸";
  if (str === "italy" || (str.includes("serie") && !str.includes("brazil"))) return "🇮🇹";
  if (str === "germany" || str.includes("bundesliga")) return "🇩🇪";
  if (str === "france" || str.includes("ligue")) return "🇫🇷";
  if (str === "netherlands" || str.includes("eredivisie")) return "🇳🇱";
  if (str === "portugal" || str.includes("liga portugal")) return "🇵🇹";
  if (str === "brazil") return "🇧🇷";
  if (str === "russia") return "🇷🇺";
  if (str === "scotland" || (str.includes("premiership") && !str.includes("betway"))) return "🏴󠁧󠁢󠁳󠁣󠁴󠁿";
  if (str === "poland" || str.includes("ekstraklasa")) return "🇵🇱";
  if (str === "belgium" || str.includes("jupiler")) return "🇧🇪";
  if (str === "turkey" || str.includes("super-lig") || str.includes("super lig")) return "🇹🇷";
  if (str === "greece") return "🇬🇷";
  if (str === "switzerland") return "🇨🇭";
  if (str === "sweden" || str.includes("allsvenskan")) return "🇸🇪";
  if (str === "norway" || str.includes("eliteserien")) return "🇳🇴";
  if (str === "denmark" || str.includes("super-liga")) return "🇩🇰";
  if (str === "mexico" || str.includes("liga-mx") || str.includes("liga mx")) return "🇲🇽";
  if (str === "usa" || str.includes("mls")) return "🇺🇸";
  if (str === "japan" || str.includes("j-1") || str.includes("j1")) return "🇯🇵";
  if (str === "china") return "🇨🇳";
  if (str === "argentina") return "🇦🇷";
  if (str === "romania") return "🇷🇴";
  if (str === "ireland" || str.includes("premier-division")) return "🇮🇪";

  return "⚽";
}

export function getCountryPriority(country: string): number {
  switch (country.toLowerCase()) {
    case "south africa": return 1;
    case "england": return 2;
    case "spain": return 3;
    case "italy": return 4;
    case "germany": return 5;
    case "france": return 6;
    case "netherlands": return 7;
    case "portugal": return 8;
    case "brazil": return 9;
    case "argentina": return 10;
    case "scotland": return 11;
    default: return 50;
  }
}

/**
 * Checks if a fixture, prediction, or odds line matches the selected league filter.
 * Accurately disambiguates:
 * - "Betway Premiership" (PSL, South Africa) vs "Premier League" (EPL, England) vs "Scottish Premiership"
 * - "Serie A" (Italy) vs "Serie A" (Brazil)
 * - "Liga 1" (Portugal) vs "Liga 1" (Romania)
 */
export function matchesLeagueFilter(
  item: { league_name?: string | null; country?: string | null; league_id?: number | null },
  selectedLeague: string,
  selectedLeagueId?: number | null
): boolean {
  // A selected ID is authoritative, including mismatches and missing item IDs.
  if (selectedLeagueId != null) return item.league_id === selectedLeagueId;
  if (/^league:\d+$/.test(selectedLeague)) return item.league_id === Number(selectedLeague.slice(7));
  if (!selectedLeague || selectedLeague.toUpperCase() === "ALL") return true;
  const norm = (value: string) => value.toLowerCase().replace(/[\s_-]+/g, "");
  const aliases: Record<string, [string, string]> = {
    premierleague: ["england", "premierleague"], epl: ["england", "premierleague"],
    championship: ["england", "championship"],
    psl: ["southafrica", "betwaypremiership"], betwaypremiership: ["southafrica", "betwaypremiership"],
    premiership: ["scotland", "premiership"], scottishpremiership: ["scotland", "premiership"],
    russianpremierleague: ["russia", "premierleague"],
    laliga: ["spain", "laliga"], seriea: ["italy", "seriea"],
    bundesliga: ["germany", "bundesliga1"], bundesliga1: ["germany", "bundesliga1"],
    ligue1: ["france", "ligue1"],
  };
  const target = aliases[norm(selectedLeague)];
  if (target) return norm(item.country || "") === target[0] && norm(item.league_name || "") === target[1];
  // Unknown names are not globally unique. UI selections must carry a league ID.
  return false;
}

export function formatMatchKickoff(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "—";

  const timeStr = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  const dayStr = formatMatchDay(dateStr);
  return `${dayStr} • ${timeStr}`;
}

export function formatSASTDateTime(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "—";

  try {
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: "Africa/Johannesburg",
      weekday: "short",
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(d) + " SAST";
  } catch {
    return d.toLocaleString("en-GB", {
      weekday: "short",
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
}

export function formatMatchDay(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  const d = new Date(dateStr);
  const now = new Date();
  
  // Calculate difference in calendar days
  const dDate = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const nowDate = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const diffDays = Math.round((dDate - nowDate) / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Tomorrow";
  if (diffDays === -1) return "Yesterday";

  return d.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

/**
 * Prioritizes upcoming matches (scheduled for today or future and unsettled) sorted soonest first,
 * with past/finished matches following behind.
 */
export function sortUpcomingFirst(
  a: { match_date?: string | null; actual_result?: string | null },
  b: { match_date?: string | null; actual_result?: string | null }
): number {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const ta = a.match_date ? new Date(a.match_date).getTime() : 0;
  const tb = b.match_date ? new Date(b.match_date).getTime() : 0;

  const aIsUpcoming = ta >= todayStart && !a.actual_result;
  const bIsUpcoming = tb >= todayStart && !b.actual_result;

  if (aIsUpcoming && !bIsUpcoming) return -1;
  if (!aIsUpcoming && bIsUpcoming) return 1;
  if (aIsUpcoming && bIsUpcoming) return ta - tb; // soonest game first
  return tb - ta; // past matches: most recent first
}

