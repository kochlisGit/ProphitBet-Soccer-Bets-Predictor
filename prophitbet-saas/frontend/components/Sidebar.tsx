"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn, planBadgeColor } from "@/lib/utils";
import { useAuth } from "@/lib/auth";

export interface NavLinkItem {
  href: string;
  label: string;
  icon: string;
  badge?: string;
}

export const navLinks: NavLinkItem[] = [
  {
    href: "/dashboard",
    label: "Dashboard",
    icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6",
  },
  {
    href: "/predictions",
    label: "Predictions",
    icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z",
  },
  {
    href: "/results",
    label: "Results & Accuracy",
    badge: "FACTUAL",
    icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
  },
  {
    href: "/mirofish",
    label: "MiroFish AI",
    badge: "SWARM",
    icon: "M13 10V3L4 14h7v7l9-11h-7z",
  },
  {
    href: "/leagues",
    label: "Leagues",
    icon: "M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064",
  },
  {
    href: "/fixtures",
    label: "Fixtures",
    icon: "M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z",
  },
  {
    href: "/live",
    label: "Live Match Watch",
    badge: "🔴 LIVE",
    icon: "M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z",
  },
  {
    href: "/models",
    label: "Models",
    icon: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z",
  },
  {
    href: "/analysis",
    label: "Analysis",
    icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
  },
  {
    href: "/sa-markets",
    label: "SA Odds & Slips",
    badge: "🇿🇦 ZAR",
    icon: "M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
  },
  {
    href: "/betslip",
    label: "Bet Journal",
    badge: "TRACKER",
    icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01",
  },
  {
    href: "/billing",
    label: "Billing",
    icon: "M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z",
  },
  {
    href: "/settings",
    label: "Settings",
    icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z",
  },
];

export const adminLink: NavLinkItem = {
  href: "/admin",
  label: "Admin",
  icon: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z",
};

export default function Sidebar() {
  const pathname = usePathname();
  const { user } = useAuth();

  const allLinks: NavLinkItem[] = user?.is_admin ? [...navLinks, adminLink] : navLinks;

  return (
    <aside className="hidden lg:flex w-64 flex-col border-r border-zinc-700/80 bg-zinc-900/95 backdrop-blur-md sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto shrink-0 z-30">
      <nav className="flex-1 space-y-1 px-3 py-5">
        <div className="px-3 pb-2 text-[10px] font-bold uppercase tracking-wider text-zinc-400">
          Navigation
        </div>
        {allLinks.map((link) => {
          const active = pathname.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                active
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 font-semibold shadow-sm"
                  : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-100 border border-transparent"
              )}
            >
              <svg
                className={cn(
                  "w-5 h-5 flex-shrink-0 transition-colors",
                  active ? "text-emerald-400" : "text-zinc-400 group-hover:text-zinc-200"
                )}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.75}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d={link.icon} />
              </svg>
              <span>{link.label}</span>
              {link.badge ? (
                <span className="ml-auto px-1.5 py-0.5 text-[9px] font-bold rounded uppercase tracking-wider bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  {link.badge}
                </span>
              ) : active ? (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]" />
              ) : null}
            </Link>
          );
        })}
      </nav>

      {user && (
        <div className="p-3 m-3 border border-zinc-800 bg-zinc-950/60 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="truncate pr-2">
              <p className="text-xs font-medium text-zinc-300 truncate">{user.email}</p>
              <span className={`badge mt-1.5 ${planBadgeColor(user.plan)}`}>
                {user.plan.toUpperCase()}
              </span>
            </div>
            <Link
              href="/settings"
              title="Settings"
              className="p-1.5 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-md transition"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              </svg>
            </Link>
          </div>
        </div>
      )}
    </aside>
  );
}
