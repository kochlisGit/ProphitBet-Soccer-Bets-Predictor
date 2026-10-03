"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/auth";

const historyKey = "prophitbetNavigationDepth";

function NavigationButton() {
  const pathname = usePathname();
  const search = useSearchParams();
  const router = useRouter();
  const { user } = useAuth();
  const depth = useRef<number | null>(null);
  const [canGoBack, setCanGoBack] = useState(false);

  useEffect(() => {
    // Preserve Next.js history metadata. Only go back through entries this app
    // has observed, never use history.length (which includes external sites).
    const stored = window.history.state?.[historyKey];
    const nextDepth = typeof stored === "number" ? stored : depth.current === null ? 0 : depth.current + 1;
    depth.current = nextDepth;
    window.history.replaceState({ ...window.history.state, [historyKey]: nextDepth }, "");
    setCanGoBack(nextDepth > 0);
  }, [pathname, search]);

  const segments = pathname.split("/").filter(Boolean);
  const fallback = segments.length > 1
    ? `/${segments.slice(0, -1).join("/")}`
    : user && pathname !== "/dashboard" ? "/dashboard" : "/";
  const disabled = !canGoBack && fallback === pathname;

  return (
    <nav aria-label="Back navigation" className="border-b border-zinc-800 bg-zinc-950/80 px-4 py-2 sm:px-6 lg:px-8">
      <button type="button" disabled={disabled}
        onClick={() => canGoBack ? router.back() : router.push(fallback)}
        aria-label={canGoBack ? "Go back to previous page" : disabled ? "No previous page" : `Go back to ${fallback === "/" ? "home" : fallback.split("/").pop()}`}
        className="inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-zinc-300 hover:bg-zinc-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 disabled:cursor-not-allowed disabled:opacity-40">
        <span aria-hidden="true">←</span> Back
      </button>
    </nav>
  );
}

export default function BackNavigation() {
  return <Suspense fallback={<div className="h-12 border-b border-zinc-800" />}><NavigationButton /></Suspense>;
}
