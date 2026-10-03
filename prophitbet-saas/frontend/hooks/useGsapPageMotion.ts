"use client";

import { type RefObject, useLayoutEffect } from "react";

/**
 * Shared page motion hook:
 * Ensures all elements inside the page root render immediately at full brightness
 * and clears any stuck inline opacity/transform styles so pages are never dark or missing cards.
 */
export function useGsapPageMotion(rootRef: RefObject<HTMLElement | null>, _reviveKey?: string) {
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    // Immediately remove any stuck inline opacity or transform styles
    const elements = root.querySelectorAll<HTMLElement>(
      ".anim-hero-line, .anim-page-title, .anim-hero-sub, .anim-page-sub, .anim-hero-actions, .anim-page-actions, .anim-hero-note, .anim-page-note, .anim-feature-card, .anim-cta-inner, .anim-reveal-inner, .anim-reveal-section, .anim-features-section, .card"
    );
    elements.forEach((el) => {
      el.style.removeProperty("opacity");
      el.style.removeProperty("transform");
    });
  }, [rootRef, _reviveKey]);
}
