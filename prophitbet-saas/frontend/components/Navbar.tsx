"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { cn, planBadgeColor } from "@/lib/utils";
import { navLinks, adminLink } from "./Sidebar";

const navLink =
  "text-sm font-medium text-zinc-400 transition-colors hover:text-zinc-100";

export default function Navbar() {
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Close mobile menu whenever pathname changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  const allLinks = user?.is_admin ? [...navLinks, adminLink] : navLinks;

  return (
    <header className="sticky top-0 z-50 border-b border-zinc-700/80 bg-zinc-900/90 shadow-nav backdrop-blur-xl supports-[backdrop-filter]:bg-zinc-900/80">
      <div className="w-full px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Brand Logo */}
          <Link href={user ? "/dashboard" : "/"} className="flex items-center gap-2.5 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-emerald-600 text-xs font-bold text-white shadow-sm ring-1 ring-white/10 group-hover:bg-emerald-500 transition-colors">
              PB
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[15px] font-semibold tracking-tight text-zinc-50">ProphitBet</span>
              {user && (
                <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] font-medium text-zinc-400 bg-zinc-800/80 rounded border border-zinc-700/60">
                  SaaS
                </span>
              )}
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden lg:flex items-center gap-4 xl:gap-5">
            {user ? (
              <>
                <Link
                  href="/dashboard"
                  className={cn(navLink, pathname.startsWith("/dashboard") && "text-emerald-400 font-semibold")}
                >
                  Dashboard
                </Link>
                <Link
                  href="/predictions"
                  className={cn(navLink, pathname.startsWith("/predictions") && "text-emerald-400 font-semibold")}
                >
                  Predictions
                </Link>
                <Link
                  href="/results"
                  className={cn(navLink, pathname.startsWith("/results") && "text-emerald-400 font-semibold")}
                >
                  Results & Accuracy
                </Link>
                <Link
                  href="/mirofish"
                  className={cn(
                    navLink,
                    "flex items-center gap-1.5",
                    pathname.startsWith("/mirofish") ? "text-purple-400 font-semibold" : "hover:text-purple-300"
                  )}
                >
                  <span>MiroFish</span>
                  <span className="px-1 py-0.5 text-[9px] font-bold rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                    AI
                  </span>
                </Link>
                <Link
                  href="/leagues"
                  className={cn(navLink, pathname.startsWith("/leagues") && "text-emerald-400 font-semibold")}
                >
                  Leagues
                </Link>
                <Link
                  href="/fixtures"
                  className={cn(navLink, pathname.startsWith("/fixtures") && "text-emerald-400 font-semibold")}
                >
                  Fixtures
                </Link>
                <Link
                  href="/live"
                  className={cn(
                    navLink,
                    "flex items-center gap-1.5",
                    pathname.startsWith("/live") ? "text-rose-400 font-semibold" : "hover:text-rose-300"
                  )}
                >
                  <span>Live Watch</span>
                  <span className="px-1.5 py-0.2 rounded-full text-[9px] font-black bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse">
                    LIVE
                  </span>
                </Link>
                <Link
                  href="/models"
                  className={cn(navLink, pathname.startsWith("/models") && "text-emerald-400 font-semibold")}
                >
                  Models
                </Link>
                <Link
                  href="/analysis"
                  className={cn(navLink, pathname.startsWith("/analysis") && "text-emerald-400 font-semibold")}
                >
                  Analysis
                </Link>
                <Link
                  href="/billing"
                  className={cn(navLink, pathname.startsWith("/billing") && "text-emerald-400 font-semibold")}
                >
                  Billing
                </Link>
                <Link
                  href="/settings"
                  className={cn(navLink, pathname.startsWith("/settings") && "text-emerald-400 font-semibold")}
                >
                  Settings
                </Link>
                {user.is_admin && (
                  <Link
                    href="/admin"
                    className={cn(
                      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold border transition",
                      pathname.startsWith("/admin")
                        ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                        : "bg-zinc-800/80 text-amber-400 border-zinc-700/80 hover:bg-zinc-750 hover:text-amber-300"
                    )}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                    Admin
                  </Link>
                )}
                <span className={`badge ${planBadgeColor(user.plan)}`}>
                  {user.plan.toUpperCase()}
                </span>
                <button
                  type="button"
                  onClick={logout}
                  className="text-sm font-medium text-zinc-400 transition-colors hover:text-red-400"
                >
                  Sign out
                </button>
              </>
            ) : (
              <>
                <Link
                  href="/pricing"
                  className={cn(navLink, pathname.startsWith("/pricing") && "text-zinc-50")}
                >
                  Pricing
                </Link>
                <Link
                  href="/login"
                  className={cn(navLink, pathname.startsWith("/login") && "text-zinc-50")}
                >
                  Sign in
                </Link>
                <Link href="/register" className="btn-primary !px-4 !py-2 text-sm">
                  Get started
                </Link>
              </>
            )}
          </div>

          {/* Mobile / Tablet Menu Button */}
          <div className="flex lg:hidden items-center gap-3">
            {user ? (
              <>
                <span className={`badge ${planBadgeColor(user.plan)}`}>
                  {user.plan.toUpperCase()}
                </span>
                <button
                  type="button"
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                  aria-label="Toggle Navigation Menu"
                  className="p-2 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 border border-zinc-700/80 transition"
                >
                  {mobileMenuOpen ? (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  )}
                </button>
              </>
            ) : (
              <div className="flex items-center gap-2">
                <Link
                  href="/login"
                  className="text-sm font-medium text-zinc-300 hover:text-white px-2 py-1"
                >
                  Sign in
                </Link>
                <Link href="/register" className="btn-primary !px-3 !py-1.5 text-xs">
                  Get started
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Menu Drawer */}
      {user && mobileMenuOpen && (
        <div className="lg:hidden border-t border-zinc-800 bg-zinc-900/95 backdrop-blur-2xl px-4 py-4 space-y-3 shadow-2xl animate-in slide-in-from-top-2 duration-200">
          <div className="px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-zinc-400">
            Menu Navigation
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {allLinks.map((link) => {
              const active = pathname.startsWith(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                      : "text-zinc-300 hover:bg-zinc-800/80 hover:text-white"
                  )}
                >
                  <svg className={cn("w-4 h-4", active ? "text-emerald-400" : "text-zinc-400")} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={link.icon} />
                  </svg>
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </div>

          <div className="pt-3 border-t border-zinc-800/80 flex items-center justify-between">
            <div className="text-xs text-zinc-400">
              Logged in as <span className="text-zinc-200 font-medium">{user.email}</span>
            </div>
            <button
              type="button"
              onClick={() => {
                setMobileMenuOpen(false);
                logout();
              }}
              className="text-xs font-semibold text-red-400 hover:text-red-300 px-2.5 py-1.5 rounded-md hover:bg-red-500/10 border border-red-500/20 transition"
            >
              Sign out
            </button>
          </div>
        </div>
      )}
    </header>
  );
}
