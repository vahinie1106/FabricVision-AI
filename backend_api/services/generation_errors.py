"""Classify garment-generation exceptions for job metadata + UI mapping."""

from __future__ import annotations

from typing import Tuple


def classify_generation_error(exc: BaseException) -> Tuple[str, str, str]:
    """
    Return (error_type, failed_stage, log_message).

    error_type is a stable machine code for the frontend.
    failed_stage is a coarse pipeline stage.
    log_message is concise and safe to store on the job (not a full traceback).
    Never include secrets (HF tokens, cookies, Authorization headers).
    """
    name = type(exc).__name__
    msg = str(exc) or name
    # Redact accidental token-looking query fragments from hub URLs.
    safe_msg = msg
    for needle in ("hf_", "Bearer ", "token="):
        if needle.lower() in safe_msg.lower():
            safe_msg = (
                safe_msg.split(needle)[0] + needle + "***REDACTED***"
                if needle in safe_msg
                else safe_msg
            )
    lower = safe_msg.lower()
    combined = f"{name}: {safe_msg}"

    if "outofmemory" in name.lower() or (
        "cuda" in lower and ("out of memory" in lower or "oom" in lower)
    ):
        stage = "diffusion"
        if any(k in lower for k in ("encode", "t5", "clip", "prompt encoding", "prompt")):
            stage = "preencode"
        if any(k in lower for k in ("vae", "decode")):
            stage = "vae"
        if "transformer inference" in lower:
            stage = "diffusion"
        return "OUT_OF_MEMORY", stage, combined

    if any(k in lower for k in ("401", "403", "unauthorized", "gated", "access to model", "auth")):
        if any(k in lower for k in ("huggingface", "hf hub", "hub.", "gated", "token", "auth")):
            return "MODEL_AUTH_FAILED", "model_load", combined

    if any(
        k in lower
        for k in (
            "model not found",
            "no flux",
            "weights missing",
            "entry not found",
            "404",
            "does not appear to have",
            "repository not found",
        )
    ):
        return "MODEL_NOT_FOUND", "model_load", combined

    if any(
        k in lower
        for k in (
            "download",
            "connection",
            "timed out",
            "timeout",
            "max retries",
            "network",
            "huggingface.co",
            "snapshot_download",
            "cannot reach",
        )
    ) and any(k in lower for k in ("model", "hub", "weight", "flux", "hf", "repo", "download")):
        return "MODEL_DOWNLOAD_FAILED", "model_load", combined

    if any(
        k in lower
        for k in (
            "fluxkontextpipeline",
            "no module named",
            "import error",
            "bitsandbytes",
            "diffusers",
            "dependency",
        )
    ) and any(k in lower for k in ("import", "module", "unavailable", "bitsandbytes", "diffusers")):
        return "MODEL_DEPENDENCY_ERROR", "model_load", combined

    if "getcurrentstream" in lower or ("device" in lower and "mismatch" in lower):
        return "DEVICE_MISMATCH", "diffusion", combined

    if (
        "no available kernel" in lower
        or "attention_kernel_unavailable" in lower
        or (
            "scaled_dot_product_attention" in lower
            and "aborting execution" in lower
        )
    ):
        return "ATTENTION_KERNEL_UNAVAILABLE", "diffusion", combined

    if "cuda" in lower and any(k in lower for k in ("error", "invalid", "fail")):
        return "CUDA_ERROR", "diffusion", combined

    if any(k in lower for k in ("bfloat16", "float16", "dtype", "half")) and any(
        k in lower for k in ("cpu", "not implemented", "expected", "mismatch")
    ):
        return "DTYPE_ERROR", "diffusion", combined

    if any(
        k in lower
        for k in (
            "failed to load",
            "unable to construct",
            "pipeline not initialized",
            "weights",
            "model_load",
            "huggingface",
            "flux.1-kontext failed to load",
        )
    ):
        return "MODEL_LOAD_ERROR", "model_load", combined

    if any(k in lower for k in ("encode_prompt", "prompt encoding", "tokenizer", "t5", "clip")):
        return "ENCODING_ERROR", "preencode", combined

    if any(k in lower for k in ("vae", "decode", "decoder")):
        return "VAE_ERROR", "vae", combined

    if any(k in lower for k in ("save", "permission", "disk", "no space")):
        return "IMAGE_SAVE_ERROR", "save", combined

    if any(k in lower for k in ("nf4", "4bit", "transformer", "diffusion", "unet")):
        return "DIFFUSION_ERROR", "diffusion", combined

    if "flux" in lower or "inference" in lower or "pipeline" in lower:
        return "PIPELINE_ERROR", "diffusion", combined

    return "GENERATION_ERROR", "unknown", combined
