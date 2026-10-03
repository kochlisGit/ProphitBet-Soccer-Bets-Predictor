"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowPathIcon, ArrowRightIcon, CheckIcon, ChevronDownIcon, PlusIcon,
  ShieldCheckIcon, SparklesIcon, TicketIcon, TrashIcon, XMarkIcon,
} from "@heroicons/react/24/outline";
import { MARKET_OPTIONS, MODEL_SELECTIONS, marketName } from "@/lib/markets";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/lib/auth";
import { saMarketsApi, betslipApi } from "@/lib/api";
import { formatSASTDateTime } from "@/lib/utils";

const markets = { H: "odds_home", D: "odds_draw", A: "odds_away", "Over 1.5": "over_15", "Under 1.5": "under_15", "Over 2.5": "over_25", "Under 2.5": "under_25", "Over 3.5": "over_35", "Under 3.5": "under_35", "BTTS Yes": "btts_yes", "BTTS No": "btts_no", "1X": "dc_1x", "12": "dc_12", "X2": "dc_x2", "DNB 1": "dnb_home", "DNB 2": "dnb_away" } as const;
type Book = "HOLLYWOODBETS" | "BETWAY";
type Mode = "ai" | "quotes" | "manual";
type Leg = { match_title: string; selection: string; odds_taken: number; bookmaker: string; quote_id?: string | null; prediction_id?: string; match_date?: string };
type Quote = Partial<Record<(typeof markets)[keyof typeof markets], number | null>> & { quote_id: string; source_url: string; scraped_at: string };
type Match = { fixture_id: string; match_title: string; league_name: string; match_date: string; bookmakers: Record<string, Quote | undefined> };
type Pick = Omit<Leg, "odds_taken"> & { market_type?: string; fixture_id: string; odds_taken: number | null; explanation: string; model_probability: number; prediction_created_at: string; source_url: string };
type Result = { legs: Pick[]; message: string; warning: string; can_save: boolean; estimated_win_probability: number | null; combined_odds: number | null };
const bookName = (book: string) => book === "BETWAY" ? "Betway South Africa" : "Hollywoodbets";
const selectionName = (selection: string) => ({ H: "Home win", D: "Draw", A: "Away win" }[selection] || selection);
const cash = (value: number) => new Intl.NumberFormat("en-ZA", { style: "currency", currency: "ZAR" }).format(value);
const field = "w-full rounded-xl border border-zinc-700 bg-zinc-950/70 px-3.5 py-3 text-sm text-white outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/20 disabled:opacity-40";
const button = "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-400 disabled:cursor-not-allowed disabled:opacity-40";
const primary = `${button} bg-emerald-400 text-zinc-950 hover:bg-emerald-300`;
const secondary = `${button} border border-zinc-700 bg-zinc-800/70 text-zinc-200 hover:bg-zinc-700`;

export default function SouthAfricanMarkets() {
  const { token, user } = useAuth();
  const [book, setBook] = useState<Book>("HOLLYWOODBETS");
  const [mode, setMode] = useState<Mode>("ai");
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [checkingFeed, setCheckingFeed] = useState(false);
  const [clock, setClock] = useState(Date.now());
  const [notice, setNotice] = useState<{ text: string; error?: boolean } | null>(null);
  const [legs, setLegs] = useState<Leg[]>([]);
  const [stake, setStake] = useState("");
  const [saving, setSaving] = useState(false);
  const [building, setBuilding] = useState(false);
  const [count, setCount] = useState(2);
  const [minimum, setMinimum] = useState(50);
  const [marketType, setMarketType] = useState("result");
  const [strategy, setStrategy] = useState("confidence");
  const [draft, setDraft] = useState(false);
  const [autoFill, setAutoFill] = useState(true);
  const [result, setResult] = useState<Result | null>(null);
  const [prices, setPrices] = useState<Record<string, string>>({});
  const [title, setTitle] = useState("");
  const [selection, setSelection] = useState("H");
  const [price, setPrice] = useState("");
  const request = useRef(0);
  const error = (e: unknown, fallback: string) => setNotice({ text: e instanceof Error ? e.message : fallback, error: true });
  const load = useCallback(async () => {
    const id = ++request.current;
    setLoading(true);
    try {
      const response = await saMarketsApi.getOdds(undefined, 100, token);
      if (id === request.current) { setMatches(response.matches || []); setClock(Date.now()); }
    } catch (e) { if (id === request.current) { setMatches([]); error(e, "Unable to load bookmaker quotes. Please retry."); } }
    finally { if (id === request.current) setLoading(false); }
  }, [token]);
  useEffect(() => { void load(); return () => { request.current += 1; }; }, [load]);
  useEffect(() => { const timer = setInterval(() => setClock(Date.now()), 30000); return () => clearInterval(timer); }, []);
  useEffect(() => {
    const timer = setInterval(() => { if (document.visibilityState === 'visible') void load(); }, 60000);
    return () => clearInterval(timer);
  }, [load]);

  const quotedMatches = matches.filter(match => {
    const quote = match.bookmakers[book.toLowerCase()];
    return quote && Date.parse(quote.scraped_at) <= clock && Date.parse(quote.scraped_at) >= clock - 15 * 60000
      && Date.parse(match.match_date) > clock && Object.values(markets).some(key => Number.isFinite(quote[key]) && Number(quote[key]) > 1);
  });
  const hasMatch = (match: string) => legs.some(leg => leg.match_title.trim().toLowerCase() === match.trim().toLowerCase());
  const add = (leg: Leg) => {
    if (saving) return false;
    if (!leg.match_title.trim() || !Number.isFinite(leg.odds_taken) || leg.odds_taken <= 1 || leg.odds_taken > 10000) {
      setNotice({ text: "Enter a match and decimal odds above 1.00 (maximum 10,000).", error: true }); return false;
    }
    if (legs.length >= 20 || hasMatch(leg.match_title) || legs.some(item => item.bookmaker !== leg.bookmaker)) {
      setNotice({ text: "Use up to 20 different events from one bookmaker. This match may already be on your slip.", error: true }); return false;
    }
    setLegs(current => [...current, leg]); setNotice(null); return true;
  };
  const generate = async (predictionOnly = draft) => {
    if (!token || building || saving) return;
    setBuilding(true); setResult(null); setPrices({}); setNotice(null);
    if (predictionOnly) { setDraft(true); setStrategy('confidence'); }
    try {
      const generated: Result = await saMarketsApi.buildAISlip({ bookmaker: book, legs: count, min_probability: minimum / 100, strategy: predictionOnly ? 'confidence' : strategy, draft: predictionOnly, market_type: marketType }, token);
      setResult(generated);
      if (autoFill && !predictionOnly && generated.can_save && generated.legs.length === count
          && generated.legs.every(pick => pick.quote_id && Number.isFinite(pick.odds_taken) && Number(pick.odds_taken) > 1)) {
        const priced = generated.legs.map(pick => ({ match_title: pick.match_title, match_date: pick.match_date,
          selection: pick.selection, odds_taken: pick.odds_taken!, bookmaker: book,
          quote_id: pick.quote_id, prediction_id: pick.prediction_id }));
        setLegs(current => current.length === 0 ? priced : current);
      }
      await load();
    }
    catch (e) { error(e, "Could not generate picks. Please retry."); }
    finally { setBuilding(false); }
  };
  const combined = legs.reduce((value, leg) => value * leg.odds_taken, 1);
  const stakeValue = Number(stake);
  const validStake = /^(?:\d+(?:\.\d{1,2})?|\.\d{1,2})$/.test(stake) && stakeValue >= .01 && stakeValue <= 1000000;
  const valid = legs.length > 0 && combined <= 1000000 && validStake;
  const save = async () => {
    if (!token || !valid || saving) return;
    setSaving(true); setNotice(null);
    try {
      await betslipApi.createSlip({ legs, stake_amount: stakeValue, notes: "Saved in ZAR from the slip studio. No bet placed." }, token);
      setLegs([]); setStake(""); setNotice({ text: "Slip saved to your journal. No bet has been placed." });
    } catch (e) { error(e, "Could not save. Refresh any stale or changed quotes."); }
    finally { setSaving(false); }
  };
  const resetResult = () => { setResult(null); setPrices({}); };

  return <DashboardLayout>
    <div className="mx-auto max-w-7xl space-y-6 pb-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div><p className="mb-2 text-[11px] font-bold uppercase tracking-[.22em] text-emerald-400">ProphitBet / Slip studio</p>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Build your next slip<span className="text-emerald-400">.</span></h1>
          <p className="mt-2 text-sm text-zinc-400">Start with AI picks. Make it yours. Review every selection.</p></div>
        <Link href="/betslip" className={secondary}><TicketIcon className="h-4 w-4"/>Open journal<ArrowRightIcon className="h-4 w-4"/></Link>
      </header>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-zinc-800 bg-zinc-950/40 px-4 py-3 sm:px-5">
        <ol aria-label="Slip building steps" className="flex flex-wrap items-center gap-3 text-xs sm:gap-5 sm:text-sm">
          {['Configure', 'Choose picks', 'Review & save'].map((label, index) => <li key={label} className="flex items-center gap-2">
            <span className={`flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-bold ${index === (legs.length ? 2 : result ? 1 : 0) ? 'bg-emerald-400 text-zinc-950' : 'bg-zinc-800 text-zinc-400'}`}>{index + 1}</span>
            <span className={index === (legs.length ? 2 : result ? 1 : 0) ? 'text-white' : 'text-zinc-500'}>{label}</span>
          </li>)}
        </ol>
        <span className="flex items-center gap-2 text-xs text-zinc-400"><ShieldCheckIcon className="h-4 w-4 text-zinc-500"/>Review only. Never auto-placed.</span>
      </div>

      {notice && <div role={notice.error ? 'alert' : 'status'} className={`flex items-start justify-between gap-3 rounded-xl border p-4 text-sm ${notice.error ? 'border-amber-500/25 bg-amber-500/5 text-amber-200' : 'border-emerald-500/25 bg-emerald-500/5 text-emerald-200'}`}>
        <span>{notice.text}</span><button aria-label="Dismiss notification" className="rounded p-1 hover:bg-white/10" onClick={() => setNotice(null)}><XMarkIcon className="h-4 w-4"/></button>
      </div>}

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_350px]">
        <div className="min-w-0 space-y-5">
          <section className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/80">
            <div className="space-y-4 border-b border-zinc-800 p-5 sm:p-6">
              <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-sm font-semibold text-white">Your bookmaker</h2><span className={`flex items-center gap-1.5 text-xs ${quotedMatches.length ? 'text-emerald-300' : 'text-zinc-400'}`}><span className={`h-1.5 w-1.5 rounded-full ${quotedMatches.length ? 'bg-emerald-400' : 'bg-amber-400'}`}/>{loading ? 'Checking quotes…' : quotedMatches.length ? `${quotedMatches.length} matches with fresh quotes` : 'Live quotes unavailable'}</span></div>
              <div className="grid grid-cols-2 gap-3">
                {(['HOLLYWOODBETS', 'BETWAY'] as Book[]).map(value => <button key={value} aria-pressed={book === value} disabled={building || saving || (legs.length > 0 && book !== value)} onClick={() => { setBook(value); resetResult(); }} className={`flex items-center gap-3 rounded-xl border p-3 text-left transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-400 disabled:opacity-40 sm:p-4 ${book === value ? 'border-emerald-400/60 bg-emerald-400/5' : 'border-zinc-700/70 hover:border-zinc-500'}`}>
                  <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm font-bold ${value === 'HOLLYWOODBETS' ? 'bg-violet-400/15 text-violet-300' : 'bg-emerald-400/15 text-emerald-300'}`}>{value === 'HOLLYWOODBETS' ? 'H' : 'B'}</span>
                  <span title={bookName(value)} className="min-w-0 flex-1 break-words text-xs font-semibold sm:text-sm">{value === 'BETWAY' ? 'Betway ZA' : 'Hollywoodbets'}</span>{book === value && <CheckIcon className="hidden h-4 w-4 shrink-0 text-emerald-400 sm:block"/>}
                </button>)}
              </div>
              {legs.length > 0 && <p className="text-xs text-zinc-500">Remove your selections before switching bookmaker.</p>}
            </div>

            <div className="p-5 sm:p-6">
              <div aria-label="Selection source" className="mb-6 grid grid-cols-3 gap-1 rounded-xl bg-zinc-950/60 p-1">
                {([{ id: 'ai', label: 'AI picks', icon: SparklesIcon }, { id: 'quotes', label: 'Live quotes', icon: ArrowPathIcon }, { id: 'manual', label: 'Manual', icon: PlusIcon }] as const).map(tab => <button key={tab.id} aria-pressed={mode === tab.id} onClick={() => setMode(tab.id)} className={`${button} px-2 py-2.5 text-xs sm:text-sm ${mode === tab.id ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-200'}`}><tab.icon className="h-4 w-4"/>{tab.label}</button>)}
              </div>

              {mode === 'ai' && <div>
                {!draft && !loading && quotedMatches.length === 0 && <p role="status" className="mb-4 rounded-xl border border-amber-400/20 bg-amber-400/5 p-4 text-xs leading-relaxed text-amber-200">No fresh verified prices are available for {bookName(book)}. Auto-picking will not invent odds or switch to a prediction-only draft. An accessible bookmaker odds source must be connected before priced picks can be generated.</p>}
                <div className="mb-6"><h2 className="text-lg font-semibold">Let the models find your picks</h2><p className="mt-1 text-sm leading-relaxed text-zinc-400">Automatically rank house-model predictions against observed bookmaker prices. Quotes refresh every minute; unavailable prices are never estimated.</p></div>
                <fieldset disabled={building || saving} className="space-y-6">
                  <label className="block text-xs text-zinc-300">Prediction market<select className={field} value={marketType} onChange={e => { setMarketType(e.target.value); resetResult(); }}>
                    <option value="all">All trained markets</option>
                    {MARKET_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select><span className="mt-2 block text-zinc-500">Needs a trained house model for this market. Verified mode also requires matching observed odds.</span></label>
                  <div><p id="leg-count-label" className="mb-3 text-xs font-medium text-zinc-300">Number of legs</p><div aria-labelledby="leg-count-label" className="grid grid-cols-6 gap-2">{[1,2,3,4,5,6].map(n => <button key={n} aria-label={`${n} ${n === 1 ? 'leg' : 'legs'}`} aria-pressed={count === n} onClick={() => { setCount(n); resetResult(); }} className={`rounded-xl border py-3 text-sm font-bold transition disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-400 ${count === n ? 'border-emerald-400 bg-emerald-400 text-zinc-950' : 'border-zinc-700 text-zinc-400 hover:border-zinc-500'}`}>{n}</button>)}</div></div>
                  <div className="grid gap-5 sm:grid-cols-2">
                    <div><label htmlFor="minimum-probability" className="flex justify-between gap-2 text-xs font-medium text-zinc-300">Minimum model probability<span className="text-emerald-300">{minimum}%</span></label><input id="minimum-probability" type="range" min="35" max="90" step="5" value={minimum} onChange={e => { setMinimum(Number(e.target.value)); resetResult(); }} className="mt-4 w-full accent-emerald-400"/><div className="mt-1 flex justify-between text-[10px] text-zinc-500"><span>More candidates · 35%</span><span>Stricter · 90%</span></div></div>
                    <div><label htmlFor="ranking" className="mb-2 block text-xs font-medium text-zinc-300">Selection priority</label><select id="ranking" className={field} value={strategy} onChange={e => { setStrategy(e.target.value); resetResult(); }}><option value="confidence">Highest model probability</option><option value="value" disabled={draft}>Estimated value · minimum 3% EV</option></select></div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">{[{ draft: true, title: 'Prediction draft', note: 'Add real odds yourself. Works without a feed.' }, { draft: false, title: 'With verified odds', note: 'Only fresh quotes from your bookmaker.' }].map(item => <button key={item.title} aria-pressed={draft === item.draft} onClick={() => { setDraft(item.draft); setStrategy('confidence'); resetResult(); }} className={`rounded-xl border p-4 text-left transition disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-400 ${draft === item.draft ? 'border-emerald-400/50 bg-emerald-400/5' : 'border-zinc-800 hover:border-zinc-600'}`}><span className="flex items-center justify-between text-sm font-semibold">{item.title}{draft === item.draft && <CheckIcon className="h-4 w-4 text-emerald-300"/>}</span><span className="mt-1 block text-xs leading-relaxed text-zinc-400">{item.note}</span></button>)}</div>
                  <label className="flex items-start gap-3 text-sm text-zinc-300"><input type="checkbox" checked={autoFill} disabled={draft} onChange={e => setAutoFill(e.target.checked)} className="mt-1 accent-emerald-400"/><span>Automatically fill an empty slip<span className="mt-1 block text-xs text-zinc-500">Verified prices only. Existing selections are kept; saving and placing bets are never automatic.</span></span></label>
                  <button onClick={() => void generate()} disabled={!token || building || saving} className={`${primary} w-full`}>{building ? <ArrowPathIcon className="h-4 w-4 animate-spin"/> : <SparklesIcon className="h-4 w-4"/>}{building ? 'Checking fixtures and ranking picks…' : !draft && autoFill ? `Auto-pick ${count} verified ${count === 1 ? 'leg' : 'legs'}` : `Generate ${count === 1 ? 'a pick' : `${count} picks`}`}{!building && <ArrowRightIcon className="ml-auto h-4 w-4"/>}</button>
                </fieldset>
                <p className="mt-3 text-center text-[11px] leading-relaxed text-zinc-500">Regulation-time markets only. No repeated teams or same-event combinations. Predictions are estimates, not guarantees.</p>
              </div>}

              {mode === 'quotes' && <div className="space-y-4">
                <div className="flex items-center justify-between gap-3"><h2 className="font-semibold">Bookmaker quotes</h2><button onClick={load} disabled={loading} className={`${secondary} px-3 py-2 text-xs`}><ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`}/>Refresh</button></div>
                {loading ? <p role="status" className="py-10 text-center text-sm text-zinc-400">Checking available prices…</p> : quotedMatches.length === 0 ? <div className="rounded-xl border border-dashed border-zinc-700 px-5 py-9 text-center"><ShieldCheckIcon className="mx-auto mb-3 h-8 w-8 text-zinc-500"/><h3 className="text-sm font-semibold">No verified quotes right now</h3><p className="mx-auto mt-2 max-w-sm text-xs leading-relaxed text-zinc-400">We don’t replace missing bookmaker prices with estimates. Start an AI draft or enter odds from your own receipt.</p><button disabled={building || saving} onClick={() => { setMode('ai'); setDraft(true); setStrategy('confidence'); resetResult(); }} className={`${secondary} mt-5 text-xs`}>Build an AI draft<ArrowRightIcon className="h-4 w-4"/></button></div> : quotedMatches.map(match => {
                  const quote = match.bookmakers[book.toLowerCase()]!;
                  return <article key={match.fixture_id} className="rounded-xl border border-zinc-800 p-4"><h3 className="text-sm font-semibold">{match.match_title}</h3><p className="mt-1 text-xs text-zinc-500">{match.league_name} · {formatSASTDateTime(match.match_date)}</p><div className="mt-4 flex flex-wrap gap-2">{Object.entries(markets).map(([pick, key]) => { const odds = quote[key]; return typeof odds === 'number' && odds > 1 && Number.isFinite(odds) && <button key={pick} disabled={saving || hasMatch(match.match_title)} onClick={() => add({ match_title: match.match_title, match_date: match.match_date, selection: pick, odds_taken: odds, bookmaker: book, quote_id: quote.quote_id })} className={`${secondary} px-3 py-2 text-xs`}>{selectionName(pick)}<span className="text-emerald-300">{odds.toFixed(2)}</span></button>; })}</div><a href={quote.source_url} target="_blank" rel="noreferrer" className="mt-3 inline-block text-[11px] text-zinc-400 underline">Observed {formatSASTDateTime(quote.scraped_at)} · source</a></article>;
                })}
                {user?.is_admin && <button disabled={checkingFeed} className="text-xs text-zinc-400 underline disabled:opacity-40" onClick={async () => { setCheckingFeed(true); try { const response = await saMarketsApi.syncOdds(token); setNotice({ text: response.message || response.status }); await load(); } catch (e) { error(e, 'Feed check failed.'); } finally { setCheckingFeed(false); } }}>{checkingFeed ? 'Checking connection…' : 'Admin: check feed connection'}</button>}
              </div>}

              {mode === 'manual' && <form className="space-y-4" onSubmit={e => { e.preventDefault(); if (add({ match_title: title.trim(), selection, odds_taken: Number(price), bookmaker: book })) { setTitle(''); setPrice(''); } }}>
                <div><h2 className="font-semibold">Add a selection from your receipt</h2><p className="mt-1 text-xs leading-relaxed text-zinc-400">Manual odds are recorded as unverified. Unlinked matches need manual settlement.</p></div>
                <fieldset disabled={saving} className="space-y-4"><label className="block text-xs text-zinc-300">Match<input required maxLength={250} className={`${field} mt-2`} value={title} placeholder="Home team vs Away team" onChange={e => setTitle(e.target.value)}/></label><div className="grid grid-cols-2 gap-3"><label className="text-xs text-zinc-300">Selection<select className={`${field} mt-2`} value={selection} onChange={e => setSelection(e.target.value)}>{[...new Set([...Object.keys(markets), ...MODEL_SELECTIONS])].map(pick => <option key={pick} value={pick}>{selectionName(pick)}</option>)}</select></label><label className="text-xs text-zinc-300">Decimal odds<input required type="number" min="1.01" max="10000" step="0.01" className={`${field} mt-2`} value={price} placeholder="e.g. 2.10" onChange={e => setPrice(e.target.value)}/></label></div><button className={`${primary} w-full`} type="submit"><PlusIcon className="h-4 w-4"/>Add to slip</button></fieldset>
              </form>}
            </div>
          </section>

          {mode === 'ai' && <section aria-busy={building} className="space-y-3">
            <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Your suggested picks</h2>{result && <span className="text-xs text-zinc-500">{result.legs.length} of {count} requested</span>}</div>
            {building ? <div role="status" className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-8 text-center"><ArrowPathIcon className="mx-auto h-6 w-6 animate-spin text-emerald-400"/><p className="mt-3 text-sm text-zinc-300">Finding matches that meet your filters</p><p className="mt-1 text-xs text-zinc-500">Verified picks can fill an empty slip. Nothing is saved or placed automatically.</p></div> : !result ? <div className="rounded-2xl border border-dashed border-zinc-800 p-8 text-center"><SparklesIcon className="mx-auto h-7 w-7 text-zinc-600"/><p className="mt-3 text-sm text-zinc-400">Your shortlist starts here</p><p className="mt-1 text-xs text-zinc-500">Set your preferences above, then generate your picks.</p></div> : <>
              {!result.legs.length && <div role="status" className="rounded-2xl border border-zinc-800 p-6"><h3 className="text-sm font-semibold">{!draft && !quotedMatches.length ? 'Bookmaker odds are unavailable' : 'No picks match these settings'}</h3><p className="mt-2 text-xs leading-relaxed text-zinc-400">{!draft && !quotedMatches.length ? 'Verified mode needs fresh bookmaker prices. Prediction picks can still be available, but they have no attached odds.' : result.message}</p>{!draft ? <button disabled={building || saving} onClick={() => void generate(true)} className={`${secondary} mt-4`}>Show prediction picks · enter odds yourself</button> : <p className="mt-3 text-xs text-zinc-400">Try a lower minimum model probability. If none qualify, an admin needs to refresh fixtures and generate predictions; both must be less than 48 hours old.</p>}</div>}
              {result.legs.map((pick, index) => { const added = hasMatch(pick.match_title); const entered = prices[pick.fixture_id] || ''; return <article key={pick.fixture_id} className={`rounded-2xl border bg-zinc-900/70 p-5 ${added ? 'border-emerald-400/30' : 'border-zinc-800'}`}>
                <div className="mb-3 flex items-center justify-between gap-3"><span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">Pick {String(index + 1).padStart(2, '0')} · {formatSASTDateTime(pick.match_date)}</span><span className={`shrink-0 rounded-full px-2.5 py-1 text-[10px] font-semibold ${added ? 'bg-emerald-400/10 text-emerald-300' : 'bg-zinc-800 text-zinc-400'}`}>{added ? 'On your slip' : pick.odds_taken == null ? 'Needs your odds' : 'Observed quote'}</span></div>
                <h3 className="text-base font-semibold text-white">{pick.match_title}</h3>
                <div className="mt-3 flex items-center justify-between rounded-xl bg-zinc-950/60 px-4 py-3"><div><p className="text-[10px] uppercase tracking-wider text-zinc-500">{marketName(pick.market_type)}</p><p className="mt-1 text-sm font-semibold text-emerald-300">{selectionName(pick.selection)}</p></div><div className="text-right"><p className="text-xl font-semibold tabular-nums">{(pick.model_probability * 100).toFixed(1)}<span className="text-xs text-zinc-500">%</span></p><p className="text-[10px] text-zinc-500">Model estimate</p></div></div>
                <details className="mt-3 text-xs text-zinc-400"><summary className="cursor-pointer py-1 hover:text-white">Why this pick?</summary><p className="mt-2 leading-relaxed">{pick.explanation}</p><p className="mt-2 text-[11px] text-zinc-500">Prediction generated {formatSASTDateTime(pick.prediction_created_at)} · <a href={pick.source_url} target="_blank" rel="noreferrer" className="underline">View source</a></p></details>
                <form className="mt-4 flex flex-wrap items-end gap-3" onSubmit={e => { e.preventDefault(); add({ match_title: pick.match_title, match_date: pick.match_date, selection: pick.selection, odds_taken: pick.odds_taken ?? Number(entered), bookmaker: book, quote_id: pick.quote_id, prediction_id: pick.prediction_id }); }}>
                  {pick.odds_taken == null ? <label className="min-w-0 flex-1 text-[11px] text-zinc-400">Your bookmaker odds · unverified<input aria-label={`Odds for ${pick.match_title}`} required type="number" min="1.01" max="10000" step="0.01" disabled={added || saving} value={entered} placeholder="Enter actual odds" onChange={e => setPrices(current => ({ ...current, [pick.fixture_id]: e.target.value }))} className={`${field} mt-1.5`}/></label> : <p className="flex-1 text-sm text-zinc-400">Observed odds <strong className="ml-2 text-lg text-white">{pick.odds_taken.toFixed(2)}</strong></p>}
                  <button type="submit" disabled={added || saving} className={added ? secondary : primary}>{added ? <CheckIcon className="h-4 w-4"/> : <PlusIcon className="h-4 w-4"/>}{added ? 'Added' : 'Add to slip'}</button>
                </form>
              </article>; })}
              {result.legs.length > 0 && <div className="rounded-xl border border-zinc-800 p-4 text-xs leading-relaxed text-zinc-400"><p>Estimated chance all suggested legs win: <span className="font-semibold text-zinc-200">{((result.estimated_win_probability || 0) * 100).toFixed(1)}%</span>. Assumes independence; not measured accuracy.</p><p className="mt-2">{result.warning}</p>{result.legs.length < count && <p className="mt-2 text-amber-200">Only {result.legs.length} selections qualify. Your filters have not been relaxed.</p>}</div>}
              {result.can_save && <button disabled={legs.length > 0 || saving} onClick={() => { setLegs(result.legs.map(pick => ({ match_title: pick.match_title, selection: pick.selection, odds_taken: pick.odds_taken!, bookmaker: book, match_date: pick.match_date, quote_id: pick.quote_id, prediction_id: pick.prediction_id }))); setNotice(null); }} className={`${secondary} w-full`}><PlusIcon className="h-4 w-4"/>{legs.length ? 'Clear current slip to add all picks' : 'Add all verified picks to slip'}</button>}
            </>}
          </section>}
        </div>

        <aside id="slip-review" aria-label="Slip review" className="scroll-mt-6 overflow-hidden rounded-2xl border border-zinc-700/80 bg-zinc-900 shadow-xl xl:sticky xl:top-6 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto">
          <div className="border-b border-zinc-800 p-5"><div className="flex items-center justify-between"><h2 className="flex items-center gap-2 font-semibold"><TicketIcon className="h-5 w-5 text-emerald-400"/>Your slip<span className="ml-1 rounded-md bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">{legs.length}</span></h2>{legs.length > 0 && <button disabled={saving} onClick={() => setLegs([])} className="rounded p-1 text-xs text-zinc-500 hover:text-zinc-200 disabled:opacity-40">Clear all</button>}</div><p className="mt-2 text-xs text-zinc-500">{bookName(book)} · {legs.length > 1 ? 'Accumulator' : 'Single'} · ZAR</p></div>
          {!legs.length ? <div className="px-7 py-10 text-center"><div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-dashed border-zinc-700"><PlusIcon className="h-6 w-6 text-zinc-600"/></div><h3 className="mt-4 text-sm font-medium text-zinc-300">A good slip starts with a pick</h3><p className="mt-2 text-xs leading-relaxed text-zinc-500">Add an AI selection, choose a verified quote, or enter your own match.</p></div> : <div className="max-h-[420px] divide-y divide-zinc-800 overflow-y-auto px-5">{legs.map((leg, index) => <div key={`${leg.match_title}-${index}`} className="py-4"><div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="break-words text-xs font-semibold leading-relaxed">{leg.match_title}</p><p className="mt-1 text-[11px] text-zinc-500">{selectionName(leg.selection)}</p></div><button aria-label={`Remove ${leg.match_title}`} disabled={saving} onClick={() => setLegs(current => current.filter((_, i) => i !== index))} className="rounded p-1 text-zinc-600 hover:text-rose-300 disabled:opacity-40"><TrashIcon className="h-4 w-4"/></button></div><div className="mt-2 flex items-center justify-between"><span className={`text-[10px] ${leg.quote_id ? 'text-emerald-400' : 'text-amber-300/80'}`}>{leg.quote_id ? 'Observed · rechecked on save' : 'Manual odds · unverified'}</span><span className="text-sm font-semibold tabular-nums">{leg.odds_taken.toFixed(2)}</span></div></div>)}</div>}
          <div className="space-y-4 border-t border-dashed border-zinc-700 bg-zinc-950/30 p-5">
            <div className="flex justify-between text-xs text-zinc-400"><span>Combined odds</span><span className="font-semibold tabular-nums text-white">{legs.length ? combined.toFixed(4) : '—'}</span></div>
            <label htmlFor="stake" className="block text-xs font-medium text-zinc-300">Your stake<div className="relative mt-2"><span className="absolute left-3.5 top-3 text-sm text-zinc-500">R</span><input id="stake" aria-describedby="stake-help" type="number" min="0.01" max="1000000" step="0.01" disabled={saving} value={stake} placeholder="0.00" onChange={e => setStake(e.target.value)} className={`${field} pl-9 text-right tabular-nums`}/></div></label>
            <p id="stake-help" className="text-[11px] text-zinc-500">Choose your own stake. No amount is suggested.</p>
            {stake && !validStake && <p role="alert" className="text-xs text-amber-200">Enter R0.01–R1,000,000 with up to two decimal places.</p>}
            {combined > 1000000 && <p role="alert" className="text-xs text-amber-200">Combined odds exceed the journal limit. Remove a selection.</p>}
            <div className="rounded-xl border border-emerald-400/15 bg-emerald-400/5 p-4"><p className="text-[11px] text-zinc-400">Potential return if every leg wins</p><p className="mt-1 break-words text-2xl font-semibold tracking-tight text-emerald-300 tabular-nums">{valid ? cash(stakeValue * combined) : '—'}</p><p className="mt-1 text-[10px] text-zinc-500">Includes stake{valid ? ` · Profit ${cash(stakeValue * (combined - 1))}` : ''}</p></div>
            <button onClick={save} disabled={!valid || !token || saving || building} className={`${primary} w-full`}>{saving ? <ArrowPathIcon className="h-4 w-4 animate-spin"/> : <TicketIcon className="h-4 w-4"/>}{saving ? 'Validating & saving…' : 'Save to journal'}</button>
            <p className="text-center text-[10px] leading-relaxed text-zinc-500">Saves a personal record, not a wager.<br/>Confirm accepted prices with your bookmaker.</p>
            <details className="border-t border-zinc-800 pt-3 text-[11px] text-zinc-500"><summary className="flex cursor-pointer items-center justify-between">How returns are calculated<ChevronDownIcon className="h-3 w-3"/></summary><p className="mt-2 leading-relaxed">Separate-event odds are multiplied. A void leg contributes 1.00. Estimates exclude boosts and bookmaker payout caps. Same-game combinations need the bookmaker’s combined price.</p></details>
          </div>
        </aside>
      </div>
      {legs.length > 0 && <a href="#slip-review" className={`${primary} fixed bottom-5 right-5 z-30 shadow-xl xl:hidden`}><TicketIcon className="h-4 w-4"/>Review slip · {legs.length}<ArrowRightIcon className="h-4 w-4"/></a>}
    </div>
  </DashboardLayout>;
}
