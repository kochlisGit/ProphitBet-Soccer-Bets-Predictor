/**
 * Load GSAP from /vendor/gsap.min.js (committed UMD build).
 * Avoids `import "gsap"` so Webpack never needs `node_modules/gsap` — fixes
 * "Can't resolve 'gsap'" when Docker volumes or installs omit the package.
 */
const SCRIPT_ID = "gsap-vendor-script";
const SCRIPT_SRC = "/vendor/gsap.min.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GsapNamespace = any;

export function loadGsap(): Promise<GsapNamespace> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("loadGsap: client only"));
  }

  const w = window as Window & { gsap?: GsapNamespace };
  if (w.gsap) {
    return Promise.resolve(w.gsap);
  }

  return new Promise((resolve, reject) => {
    const existing = document.getElementById(SCRIPT_ID) as HTMLScriptElement | null;
    const finish = () => {
      const g = (window as Window & { gsap?: GsapNamespace }).gsap;
      if (g) resolve(g);
      else reject(new Error("GSAP global missing after script load"));
    };

    if (existing) {
      if (existing.dataset.loaded === "1") {
        finish();
        return;
      }
      existing.addEventListener("load", finish);
      existing.addEventListener("error", () => reject(new Error("GSAP script load error")));
      return;
    }

    const el = document.createElement("script");
    el.id = SCRIPT_ID;
    el.src = SCRIPT_SRC;
    el.async = true;
    el.onload = () => {
      el.dataset.loaded = "1";
      finish();
    };
    el.onerror = () => reject(new Error(`Failed to load ${SCRIPT_SRC}`));
    document.head.appendChild(el);
  });
}
