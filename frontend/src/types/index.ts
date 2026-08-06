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
  error?: string;
}
