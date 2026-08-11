/** Poll /api/v1/flux-status for API-process FLUX warmup UX. */

import { ApiClient } from "./api";

export type FluxWarmupState =
  | "IDLE"
  | "STARTING"
  | "READY"
  | "FAILED"
  | "SKIPPED"
  | "UNKNOWN";

export interface FluxStatusResponse {
  state: FluxWarmupState | string;
  state_raw?: string;
  ready?: boolean;
  in_memory?: boolean;
  pipeline_exists?: boolean;
  progress?: number;
  current_step?: string;
  stage?: string;
  load_duration_s?: number | null;
  cache_status?: string | null;
  error?: string | null;
  api_pid?: number;
  pid?: number;
  model_reused?: boolean;
}

export async function fetchFluxStatus(): Promise<FluxStatusResponse> {
  return ApiClient.get<FluxStatusResponse>("/flux-status");
}
