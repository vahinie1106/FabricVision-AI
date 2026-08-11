export type ProgressStageDef = {
  id: string;
  label: string;
  /** Approximate completion percentage when this stage becomes active */
  percent: number;
};

/** Stage start percents aligned with backend garment job progress bands. */
export const GARMENT_STAGES: ProgressStageDef[] = [
  { id: "upload", label: "Uploading Fabric", percent: 5 },
  { id: "load-model", label: "Loading / preparing FLUX", percent: 8 },
  { id: "fabric", label: "Preparing fabric", percent: 22 },
  { id: "conditioning", label: "Preparing garment conditioning", percent: 32 },
  { id: "prompt", label: "Encoding prompt", percent: 42 },
  { id: "generate", label: "Generating", percent: 50 },
  { id: "decode", label: "Decoding image", percent: 88 },
  { id: "save", label: "Saving result", percent: 94 },
  { id: "done", label: "Completed", percent: 100 },
];

export const TRYON_STAGES: ProgressStageDef[] = [
  { id: "upload", label: "Uploading Images", percent: 10 },
  { id: "load", label: "Loading CatVTON", percent: 30 },
  { id: "align", label: "Aligning Garments", percent: 50 },
  { id: "generate", label: "Generating Result", percent: 75 },
  { id: "refine", label: "Refining", percent: 90 },
  { id: "done", label: "Completed", percent: 100 },
];

export const ANALYSIS_STAGES: ProgressStageDef[] = [
  { id: "upload", label: "Uploading", percent: 10 },
  { id: "load", label: "Loading Model", percent: 30 },
  { id: "analyze", label: "Analyzing", percent: 55 },
  { id: "extract", label: "Extracting Metadata", percent: 80 },
  { id: "validate", label: "Validating", percent: 95 },
  { id: "done", label: "Completed", percent: 100 },
];

export type WorkflowKind = "garment" | "tryon" | "analysis";

/** Backend JobStatusResponse.stage → frontend GARMENT_STAGES id */
export const BACKEND_STAGE_TO_UI_ID: Record<string, string> = {
  queued: "upload",
  initializing: "upload",
  loading_model: "load-model",
  preparing_fabric: "fabric",
  preparing_conditioning: "conditioning",
  encoding_prompt: "prompt",
  generating: "generate",
  decoding: "decode",
  validating: "save",
  saving: "save",
  completed: "done",
  failed: "done",
  processing: "fabric",
};

export function getStages(kind: WorkflowKind): ProgressStageDef[] {
  switch (kind) {
    case "garment":
      return GARMENT_STAGES;
    case "tryon":
      return TRYON_STAGES;
    case "analysis":
      return ANALYSIS_STAGES;
  }
}

/**
 * Maps a backend progress value / step / stage onto a stage index.
 * Prefer authoritative `stage` from GET /status when present.
 */
export function resolveStageIndex(
  stages: ProgressStageDef[],
  progress: number,
  currentStep?: string,
  backendStage?: string
): number {
  if (progress >= 100) return stages.length - 1;

  if (backendStage) {
    const uiId = BACKEND_STAGE_TO_UI_ID[backendStage];
    if (uiId) {
      const idx = stages.findIndex((s) => s.id === uiId);
      if (idx >= 0) return idx;
    }
  }

  if (currentStep) {
    const lower = currentStep.toLowerCase();
    // Ordered specific → general. Never use bare "load" (matches "uploading").
    const keywordMap: Array<{ keys: string[]; id: string }> = [
      {
        keys: [
          "reusing",
          "loading model",
          "loading flux",
          "initializing flux",
          "connecting to remote",
          "downloading flux",
          "downloading",
          "flux ready",
          "transformer",
          "pipeline",
          "offload",
          "cache hit",
          "cache miss",
          "in-memory",
        ],
        id: "load-model",
      },
      { keys: ["preparing fabric", "fabric appearance"], id: "fabric" },
      { keys: ["garment conditioning", "conditioning"], id: "conditioning" },
      { keys: ["encoding prompt"], id: "prompt" },
      { keys: ["generating", "diffusion", "step "], id: "generate" },
      { keys: ["decoding"], id: "decode" },
      { keys: ["saving"], id: "save" },
      { keys: ["completed"], id: "done" },
      { keys: ["waiting for worker", "upload"], id: "upload" },
    ];
    for (const entry of keywordMap) {
      if (entry.keys.some((k) => lower.includes(k))) {
        const idx = stages.findIndex((s) => s.id === entry.id);
        if (idx >= 0) return idx;
      }
    }
  }

  let idx = 0;
  for (let i = 0; i < stages.length; i++) {
    if (progress >= stages[i].percent) idx = i;
  }
  return Math.min(idx, stages.length - 2);
}
