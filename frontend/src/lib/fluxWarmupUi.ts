/**
 * Pure FLUX warmup UI rules for Custom Garment.
 * IDLE / UNKNOWN / poll failure must never lock the studio at "0% warming".
 */

export type FluxWarmupUiState =
  | "IDLE"
  | "STARTING"
  | "READY"
  | "FAILED"
  | "SKIPPED"
  | "UNKNOWN";

export type FluxWarmupUiInput = {
  state?: string | null;
  ready?: boolean | null;
  in_memory?: boolean | null;
  progress?: number | null;
  current_step?: string | null;
  stage?: string | null;
  error?: string | null;
  pollError?: string | null;
  load_duration_s?: number | null;
};

export type FluxWarmupUiView = {
  state: FluxWarmupUiState;
  warming: boolean;
  ready: boolean;
  failed: boolean;
  /** True when Generate should stay clickable (fabric upload still required separately). */
  generateEnabledByFlux: boolean;
  showWarmupBanner: boolean;
  progress: number;
  currentStep: string | null;
  error: string | null;
  loadDurationS: number | null;
};

export function normalizeFluxWarmupState(raw: string | undefined | null): FluxWarmupUiState {
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

export function deriveFluxWarmupUi(input: FluxWarmupUiInput): FluxWarmupUiView {
  const state = normalizeFluxWarmupState(input.state ?? undefined);
  const ready =
    state === "READY" || Boolean(input.ready) || Boolean(input.in_memory);
  // Only an explicit in-process load counts as warming.
  const warming = state === "STARTING" && !ready;
  const failed = state === "FAILED";
  const pollError = (input.pollError || "").trim() || null;
  const statusError = (input.error || "").trim() || null;
  const error = statusError || pollError;

  return {
    state,
    warming,
    ready,
    failed,
    // IDLE / UNKNOWN / temporary poll failure / FAILED (retry) must not block Generate.
    // Only an active STARTING load temporarily waits for the engine.
    generateEnabledByFlux: !warming,
    showWarmupBanner: warming,
    progress: warming
      ? Math.max(1, Math.min(100, Number(input.progress) || 1))
      : Math.max(0, Math.min(100, Number(input.progress) || 0)),
    currentStep: input.current_step || input.stage || null,
    error,
    loadDurationS:
      typeof input.load_duration_s === "number" ? input.load_duration_s : null,
  };
}
