"use client";

import { useEffect, useState } from "react";
import {
  deriveFluxWarmupUi,
  type FluxWarmupUiState,
} from "@/lib/fluxWarmupUi";
import {
  fetchFluxStatus,
  type FluxStatusResponse,
} from "@/services/fluxStatus";

/**
 * Poll API-process FLUX warmup status for studio UX.
 * Does not fake progress — mirrors /api/v1/flux-status.
 * IDLE / UNKNOWN / temporary poll failures are NOT treated as warming.
 */
export function useFluxWarmupStatus(pollMs = 2000) {
  const [status, setStatus] = useState<FluxStatusResponse | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const next = await fetchFluxStatus();
        if (cancelled) return;
        setStatus(next);
        setPollError(null);
        const state = String(next.state || "").toUpperCase();
        // Keep polling after FAILED so a Generate-driven retry can clear the banner.
        if (state === "READY" || state === "SKIPPED") {
          return;
        }
      } catch (exc) {
        if (cancelled) return;
        setPollError(exc instanceof Error ? exc.message : String(exc));
      }
      if (!cancelled) {
        timer = setTimeout(tick, pollMs);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [pollMs]);

  const ui = deriveFluxWarmupUi({
    state: status?.state ? String(status.state) : pollError ? "UNKNOWN" : undefined,
    ready: status?.ready,
    in_memory: status?.in_memory,
    progress: status?.progress,
    current_step: status?.current_step,
    stage: status?.stage,
    error: status?.error,
    pollError,
    load_duration_s: status?.load_duration_s,
  });

  return {
    status,
    state: ui.state as FluxWarmupUiState,
    warming: ui.warming,
    ready: ui.ready,
    failed: ui.failed,
    generateEnabledByFlux: ui.generateEnabledByFlux,
    showWarmupBanner: ui.showWarmupBanner,
    error: ui.error,
    progress: ui.progress,
    currentStep: ui.currentStep,
    loadDurationS: ui.loadDurationS,
  };
}
