export type ProgressStageDef = {
  id: string;
  label: string;
  /** Approximate completion percentage when this stage becomes active */
  percent: number;
};

export const GARMENT_STAGES: ProgressStageDef[] = [
  { id: "upload", label: "Uploading Fabric", percent: 5 },
  { id: "load-model", label: "Loading model", percent: 12 },
  { id: "fabric", label: "Preparing fabric", percent: 22 },
  { id: "conditioning", label: "Preparing garment conditioning", percent: 35 },
  { id: "prompt", label: "Encoding prompt", percent: 48 },
  { id: "generate", label: "Generating", percent: 65 },
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
 * Maps a backend progress value / step string onto a stage index.
 */
export function resolveStageIndex(
  stages: ProgressStageDef[],
  progress: number,
  currentStep?: string
): number {
  if (progress >= 100) return stages.length - 1;

  if (currentStep) {
    const lower = currentStep.toLowerCase();
    const keywordMap: Array<{ keys: string[]; id: string }> = [
      { keys: ["reusing", "loading model", "load"], id: "load-model" },
      { keys: ["preparing fabric", "fabric appearance"], id: "fabric" },
      { keys: ["conditioning", "garment conditioning"], id: "conditioning" },
      { keys: ["encoding prompt", "prompt"], id: "prompt" },
      { keys: ["generating", "diffusion", "step "], id: "generate" },
      { keys: ["decoding"], id: "decode" },
      { keys: ["saving"], id: "save" },
      { keys: ["completed"], id: "done" },
    ];
    for (const entry of keywordMap) {
      if (entry.keys.some((k) => lower.includes(k))) {
        const idx = stages.findIndex((s) => s.id === entry.id);
        if (idx >= 0) return idx;
      }
    }
    const byLabel = stages.findIndex(
      (s) =>
        s.id !== "done" &&
        (lower.includes(s.id) || lower.includes(s.label.toLowerCase().split(" ")[0]))
    );
    if (byLabel >= 0) return byLabel;
  }

  let idx = 0;
  for (let i = 0; i < stages.length; i++) {
    if (progress >= stages[i].percent) idx = i;
  }
  return Math.min(idx, stages.length - 2);
}
