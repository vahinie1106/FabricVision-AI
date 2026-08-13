export type FriendlyError = {
  title: string;
  message: string;
  code:
    | "GPU_BUSY"
    | "CUDA_MEMORY"
    | "MODEL_LOAD"
    | "DEVICE_MISMATCH"
    | "ENCODING_ERROR"
    | "DIFFUSION_ERROR"
    | "VAE_ERROR"
    | "IMAGE_SAVE_ERROR"
    | "PIPELINE_ERROR"
    | "GENERATION_FAILED"
    | "INVALID_IMAGE"
    | "SERVER_OFFLINE"
    | "TIMEOUT"
    | "BACKEND_RESTARTED"
    | "CATVTON_FALLBACK"
    | "CATVTON_MASK"
    | "GENERIC";
};

type ErrorExtras = {
  errorType?: string;
  failedStage?: string;
};

function extrasFrom(error: unknown): ErrorExtras {
  if (error && typeof error === "object") {
    const e = error as Record<string, unknown>;
    return {
      errorType: typeof e.errorType === "string" ? e.errorType : undefined,
      failedStage: typeof e.failedStage === "string" ? e.failedStage : undefined,
    };
  }
  return {};
}

export function toFriendlyError(error: unknown): FriendlyError {
  const raw =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "An unexpected error occurred";

  const lower = raw.toLowerCase();
  const { errorType, failedStage } = extrasFrom(error);
  const type = (errorType || "").toUpperCase();

  if (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("network request failed") ||
    lower.includes("err_connection") ||
    lower.includes("econnrefused")
  ) {
    return {
      code: "SERVER_OFFLINE",
      title: "Server Offline",
      message: "Unable to connect to the backend. Please check that the API server is running.",
    };
  }

  if (
    type === "CUDA_OOM" ||
    type === "OUT_OF_MEMORY" ||
    lower.includes("out of memory") ||
    lower.includes("outofmemory") ||
    (lower.includes("cuda") && lower.includes("out of memory"))
  ) {
    return {
      code: "CUDA_MEMORY",
      title: "CUDA Memory Full",
      message:
        "The GPU ran out of memory during generation. Wait a moment and try again — the backend frees VRAM after a failed job.",
    };
  }

  if (type === "MODEL_AUTH_FAILED" || lower.includes("model_auth_failed")) {
    return {
      code: "MODEL_LOAD",
      title: "Model Access Denied",
      message:
        "Hugging Face authentication failed while loading FLUX. Add a Kaggle secret named HF_TOKEN if the model requires access.",
    };
  }

  if (type === "MODEL_NOT_FOUND" || lower.includes("model_not_found")) {
    return {
      code: "MODEL_LOAD",
      title: "Model Not Found",
      message:
        "FLUX Kontext weights were not found. On Kaggle, allow the first run to download eramth/flux-kontext-4bit, or run scripts/download_flux_kontext.py.",
    };
  }

  if (type === "MODEL_DOWNLOAD_FAILED" || lower.includes("model_download_failed")) {
    return {
      code: "MODEL_LOAD",
      title: "Model Download Failed",
      message:
        "FLUX weights could not be downloaded from Hugging Face. Check Kaggle internet access and try again.",
    };
  }

  if (type === "MODEL_DEPENDENCY_ERROR" || lower.includes("model_dependency_error")) {
    return {
      code: "MODEL_LOAD",
      title: "Model Dependency Error",
      message:
        "A required FLUX dependency is missing or incompatible (diffusers / bitsandbytes / FluxKontextPipeline).",
    };
  }

  if (type === "CUDA_ERROR" || (lower.includes("cuda") && lower.includes("error"))) {
    return {
      code: "DEVICE_MISMATCH",
      title: "GPU Error",
      message: "A CUDA error interrupted FLUX loading or generation. Check the backend logs for details.",
    };
  }

  if (type === "DEVICE_MISMATCH" || lower.includes("getcurrentstream") || lower.includes("device mismatch")) {
    return {
      code: "DEVICE_MISMATCH",
      title: "GPU Device Error",
      message:
        "A CUDA device/offload mismatch interrupted garment generation. Please try again — the pipeline has been reset.",
    };
  }

  if (type === "MODEL_LOAD_ERROR" || lower.includes("failed to load") || lower.includes("weights missing")) {
    return {
      code: "MODEL_LOAD",
      title: "Model Load Failed",
      message:
        "The FLUX Kontext model could not be loaded. Check backend logs for the root exception (weights, auth, download, or CUDA).",
    };
  }

  if (type === "ENCODING_ERROR" || (failedStage === "preencode" && type !== "CUDA_OOM")) {
    return {
      code: "ENCODING_ERROR",
      title: "Prompt Encoding Failed",
      message: "Garment generation failed while encoding the prompt.",
    };
  }

  if (type === "VAE_ERROR" || failedStage === "vae") {
    return {
      code: "VAE_ERROR",
      title: "Image Decode Failed",
      message: "Garment image decoding failed. Please try again.",
    };
  }

  if (type === "IMAGE_SAVE_ERROR" || failedStage === "save") {
    return {
      code: "IMAGE_SAVE_ERROR",
      title: "Save Failed",
      message: "The garment was generated but could not be saved. Please try again.",
    };
  }

  if (type === "DIFFUSION_ERROR" || type === "PIPELINE_ERROR" || failedStage === "diffusion") {
    return {
      code: "DIFFUSION_ERROR",
      title: "Generation Failed",
      message: "Garment generation failed during model inference. Please try again.",
    };
  }

  if (
    type === "CATVTON_FALLBACK" ||
    lower.includes("blend_preview") ||
    lower.includes("did not complete with real catvton") ||
    lower.includes("fallback rejected")
  ) {
    return {
      code: "CATVTON_FALLBACK",
      title: "Real CatVTON Required",
      message:
        "A preview blend was blocked — FabricVision only shows real CatVTON try-on results. Check that CatVTON weights are loaded and the person mask succeeded.",
    };
  }

  if (
    type === "CATVTON_MASK_QUALITY" ||
    lower.includes("box_fallback") ||
    lower.includes("rectangular box mask") ||
    lower.includes("insufficient clothing mask")
  ) {
    return {
      code: "CATVTON_MASK",
      title: "Clothing Mask Unavailable",
      message:
        "Could not build a reliable clothing-region mask for try-on. A rectangular box mask would produce a misleading overlay, so the job was stopped.",
    };
  }

  if (type === "CATVTON_MODEL_MISSING") {
    return {
      code: "MODEL_LOAD",
      title: "CatVTON Model Missing",
      message: "CatVTON weights could not be loaded. Install models/CatVTON before running try-on.",
    };
  }

  if (
    lower.includes("gpu") &&
    (lower.includes("busy") || lower.includes("occupied") || lower.includes("in use") || lower.includes("queue"))
  ) {
    return {
      code: "GPU_BUSY",
      title: "GPU Busy",
      message: "The AI model is currently processing another request.",
    };
  }

  if (
    lower.includes("invalid image") ||
    lower.includes("unsupported") ||
    lower.includes("image format") ||
    lower.includes("corrupt")
  ) {
    return {
      code: "INVALID_IMAGE",
      title: "Invalid Image",
      message: "Please upload a valid garment image (PNG, JPEG, or WEBP).",
    };
  }

  if (
    type === "BACKEND_RESTARTED" ||
    lower.includes("backend_restarted") ||
    lower.includes("backend process restarted") ||
    lower.includes("job not found")
  ) {
    return {
      code: "BACKEND_RESTARTED",
      title: "Backend Restarted",
      message:
        "The API process restarted while this job was running. Click Generate again — the job was not lost as a silent 404.",
    };
  }

  if (lower.includes("timeout") || lower.includes("timed out")) {
    return {
      code: "TIMEOUT",
      title: "Request Timed Out",
      message: "The generation took longer than expected. Please try again.",
    };
  }

  if (type === "UNKNOWN_GENERATION_ERROR" || lower.includes("flux") || lower.includes("inference failed")) {
    return {
      code: "GENERATION_FAILED",
      title: "Generation Failed",
      message: "Garment generation failed during model inference. Please try again.",
    };
  }

  const looksTechnical =
    lower.includes("traceback") ||
    lower.includes("exception") ||
    lower.includes("runtimeerror") ||
    lower.includes("typeerror") ||
    lower.includes("filenotfound") ||
    raw.length > 180;

  return {
    code: "GENERIC",
    title: "Something Went Wrong",
    message: looksTechnical
      ? "We couldn't complete this request. Please try again in a moment."
      : raw,
  };
}
