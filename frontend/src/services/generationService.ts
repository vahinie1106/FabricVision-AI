import { ApiClient } from "./api";
import type { GenerationRequest, GenerationResponse, TryOnRequest, TryOnResponse, GenerationStatus } from "@/types";

export const GenerationService = {
  async generateCustomGarment(
    req: GenerationRequest,
    onProgress?: (progress: number, step: string) => void
  ): Promise<GenerationResponse> {
    const formData = new FormData();
    formData.append("fabric_image", req.fabricImage);
    formData.append("garment_type", req.garmentType);
    formData.append("fit", req.fit);
    formData.append("style", req.style);
    formData.append("gender", req.gender);
    formData.append("season", req.season);
    formData.append("occasion", req.occasion);
    formData.append("fabric", req.fabric);
    formData.append("material", req.material);
    formData.append("texture", req.texture);
    formData.append("color", req.color);
    formData.append("sleeve", req.sleeve);
    formData.append("neckline", req.neckline);

    const initRes = await ApiClient.postFormData<{ job_id: string }>("/generate", formData);
    
    if (!initRes.job_id) {
      throw new Error("Failed to start generation job");
    }

    const finalStatus = await ApiClient.pollJobStatus(initRes.job_id, (status) => {
      if (onProgress) {
        onProgress(status.progress, status.current_step);
      }
    });

    if (finalStatus.status === "failed") {
      throw new Error(finalStatus.error || "Generation failed");
    }

    return {
      id: finalStatus.job_id,
      status: finalStatus.status as GenerationStatus,
      resultUrl: finalStatus.result_url,
      metadata: finalStatus.metadata
    };
  },

  async executeVirtualTryOn(
    req: TryOnRequest,
    onProgress?: (progress: number, step: string) => void
  ): Promise<TryOnResponse> {
    const formData = new FormData();
    formData.append("garment_image", req.garmentImage);
    formData.append("person_image", req.personImage);
    formData.append("fit_preference", req.fitPreference);
    formData.append("background_action", req.backgroundAction);

    const initRes = await ApiClient.postFormData<{ job_id: string }>("/tryon", formData);

    if (!initRes.job_id) {
      throw new Error("Failed to start try-on job");
    }

    const finalStatus = await ApiClient.pollJobStatus(initRes.job_id, (status) => {
      if (onProgress) {
        onProgress(status.progress, status.current_step);
      }
    });

    if (finalStatus.status === "failed") {
      throw new Error(finalStatus.error || "Virtual try-on failed");
    }

    return {
      id: finalStatus.job_id,
      status: finalStatus.status as GenerationStatus,
      resultUrl: finalStatus.result_url
    };
  }
};
