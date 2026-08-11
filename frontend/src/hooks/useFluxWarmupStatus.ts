"use client";

import { useEffect, useState } from "react";
import {
  fetchFluxStatus,
  type FluxStatusResponse,
  type FluxWarmupState,
} from "@/services/fluxStatus";

function normalizeState(raw: string | undefined): FluxWarmupState {
  const s = (raw || "UNKNOWN").toUpperCase();
  if (
    s === "IDLE" ||
    s === "STARTING" ||
    s === "READY" ||
    s === "FAILED" ||
    s === "SKIPPED"
  ) {
    return s;
  }
  if (s === "LOADING") return "STARTING";
  return "UNKNOWN";
}

/**
 * Poll API-process FLUX warmup status for studio UX.
 * Does not fake progress — mirrors /api/v1/flux-status.
 */
export function useFluxWarmupStatus(pollMs = 2000) {
  const [status, setStatus] = useState<FluxStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const next = await fetchFluxStatus();
        if (cancelled) return;
        setStatus(next);
        setError(null);
        const state = normalizeState(String(next.state));
        if (state === "READY" || state === "FAILED" || state === "SKIPPED") {
          return;
        }
      } catch (exc) {
        if (cancelled) return;
        setError(exc instanceof Error ? exc.message : String(exc));
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

  const state = normalizeState(status?.state ? String(status.state) : undefined);
  const warming = state === "STARTING" || state === "IDLE" || state === "UNKNOWN";
  const ready = state === "READY" || Boolean(status?.ready) || Boolean(status?.in_memory);
  const failed = state === "FAILED";

  return {
    status,
    state,
    warming: warming && !ready,
    ready,
    failed,
    error: error || status?.error || null,
    progress: status?.progress ?? 0,
    currentStep: status?.current_step || status?.stage || null,
    loadDurationS: status?.load_duration_s ?? null,
  };
}
