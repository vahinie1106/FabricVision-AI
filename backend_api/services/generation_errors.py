"""Classify garment-generation exceptions for job metadata + UI mapping."""

from __future__ import annotations

from typing import Tuple


def classify_generation_error(exc: BaseException) -> Tuple[str, str, str]:
    """
    Return (error_type, failed_stage, log_message).

    error_type is a stable machine code for the frontend.
    failed_stage is a coarse pipeline stage.
    log_message is concise and safe to store on the job (not a full traceback).
    """
    name = type(exc).__name__
    msg = str(exc) or name
    lower = msg.lower()
    combined = f"{name}: {msg}"

    if "outofmemory" in name.lower() or (
        "cuda" in lower and ("out of memory" in lower or "oom" in lower)
    ):
        stage = "preencode" if any(k in lower for k in ("encode", "t5", "clip", "prompt")) else "diffusion"
        if "vae" in lower or "decode" in lower:
            stage = "vae"
        return "CUDA_OOM", stage, combined

    if "getcurrentstream" in lower or "device" in lower and "mismatch" in lower:
        return "DEVICE_MISMATCH", "diffusion", combined

    if any(k in lower for k in ("bfloat16", "float16", "dtype", "half")) and any(
        k in lower for k in ("cpu", "not implemented", "expected", "mismatch")
    ):
        return "DTYPE_ERROR", "diffusion", combined

    if any(k in lower for k in ("load", "weights", "pipeline not initialized", "huggingface")):
        return "MODEL_LOAD_ERROR", "model_load", combined

    if any(k in lower for k in ("encode_prompt", "prompt encoding", "tokenizer", "t5", "clip")):
        return "ENCODING_ERROR", "preencode", combined

    if any(k in lower for k in ("vae", "decode", "decoder")):
        return "VAE_ERROR", "vae", combined

    if any(k in lower for k in ("save", "permission", "disk", "no space")):
        return "IMAGE_SAVE_ERROR", "save", combined

    if any(k in lower for k in ("bitsandbytes", "nf4", "4bit", "transformer", "diffusion", "unet")):
        return "DIFFUSION_ERROR", "diffusion", combined

    if "flux" in lower or "inference" in lower or "pipeline" in lower:
        return "PIPELINE_ERROR", "diffusion", combined

    return "UNKNOWN_GENERATION_ERROR", "unknown", combined
