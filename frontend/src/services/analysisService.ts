import { ApiClient } from "./api";

export interface SemanticAnalysisRequest {
  garmentImage: File;
}

export interface SemanticAnalysisResponse {
  status: string;
  metadata?: any;
  confidence?: number;
  error?: string;
}

export const AnalysisService = {
  async analyzeGarment(req: SemanticAnalysisRequest): Promise<SemanticAnalysisResponse> {
    const formData = new FormData();
    formData.append("garment_image", req.garmentImage);

    // Endpoint is synchronous, it returns the metadata directly.
    const response = await ApiClient.postFormData<SemanticAnalysisResponse>("/analyze", formData);
    
    if (response.status === "failed") {
      throw new Error(response.error || "Analysis failed");
    }

    return response;
  }
};
