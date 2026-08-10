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
    formData.append("generation_mode", req.generationMode || "standard");

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
      const err = new Error(finalStatus.error || "Generation failed") as Error & {
        errorType?: string;
        failedStage?: string;
      };
      err.errorType = finalStatus.error_type || finalStatus.metadata?.error_type;
      err.failedStage = finalStatus.failed_stage || finalStatus.metadata?.failed_stage;
      throw err;
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
      const err = new Error(finalStatus.error || "Virtual try-on failed") as Error & {
        errorType?: string;
        failedStage?: string;
        metadata?: Record<string, unknown>;
      };
      err.errorType = finalStatus.error_type || finalStatus.metadata?.error_type;
      err.failedStage = finalStatus.failed_stage || finalStatus.metadata?.failed_stage;
      err.metadata = finalStatus.metadata;
      throw err;
    }

    const meta = finalStatus.metadata || {};
    // Defense-in-depth: never treat blend/fallback as UI success even if status slipped.
    if (meta.was_fallback_used || meta.was_real_catvton_used === false) {
      const err = new Error(
        "Virtual try-on did not use real CatVTON inference (fallback/blend rejected)."
      ) as Error & { errorType?: string; failedStage?: string; metadata?: Record<string, unknown> };
      err.errorType = "CATVTON_FALLBACK";
      err.failedStage = meta.failed_stage || "fallback";
      err.metadata = meta;
      throw err;
    }

    return {
      id: finalStatus.job_id,
      status: finalStatus.status as GenerationStatus,
      resultUrl: finalStatus.result_url,
      metadata: finalStatus.metadata,
    };
  }
};
