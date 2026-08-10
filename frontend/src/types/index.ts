// Global Types for FabricVision-AI

export type GenerationStatus = "idle" | "validating" | "processing" | "completed" | "failed";

export interface Project {
  id: string;
  name: string;
  createdAt: string;
  thumbnailUrl: string;
  moduleUsed: "custom-garment" | "virtual-tryon" | "semantic-analysis";
  status: "draft" | "completed";
}

export interface GarmentMetadata {
  category: string;
  fabric: string;
  styleAffinity: string;
  confidenceScore: number;
}

/** Garment generation quality mode (maps to backend Preview / Standard / Production). */
export type GenerationMode = "Preview" | "Standard" | "Production";

export interface GenerationRequest {
  fabricImage: File;
  garmentType: string;
  fit: string;
  style: string;
  gender: string;
  season: string;
  occasion: string;
  fabric: string;
  material: string;
  texture: string;
  color: string;
  sleeve: string;
  neckline: string;
  /** Default: Standard — balanced quality on RTX 3050 6GB */
  generationMode?: GenerationMode;
}

export interface GenerationResponse {
  id: string;
  status: GenerationStatus;
  resultUrl?: string;
  metadata?: GarmentMetadata;
  error?: string;
}

export interface TryOnRequest {
  garmentImage: File;
  personImage: File;
  fitPreference: string;
  backgroundAction: string;
}

export interface TryOnResponse {
  id: string;
  status: GenerationStatus;
  resultUrl?: string;
  metadata?: Record<string, unknown>;
  error?: string;
}

export type StudioModule = "custom-garment" | "virtual-tryon" | "semantic-analysis";

export interface ResultViewerMeta {
  model?: string;
  timestamp?: string;
  resolution?: string;
  durationMs?: number;
  promptSummary?: string;
}
