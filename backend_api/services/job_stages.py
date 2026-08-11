"""Authoritative job stage mapping for generation / try-on progress."""

from __future__ import annotations

from typing import Optional


# Canonical stages for garment generation (backend is source of truth).
GARMENT_STAGE_ORDER = (
    "queued",
    "initializing",
    "loading_model",
    "preparing_fabric",
    "encoding_prompt",
    "preparing_conditioning",
    "generating",
    "decoding",
    "validating",
    "saving",
    "completed",
    "failed",
)


def map_step_to_stage(current_step: Optional[str], status: Optional[str] = None) -> str:
    """Map a human-readable progress step onto a stable stage id."""
    if status == "failed":
        return "failed"
    if status == "completed":
        return "completed"
    if status == "queued":
        return "queued"

    lower = (current_step or "").lower()
    if not lower:
        return "processing" if status == "processing" else "queued"

    rules = (
        (("fail",), "failed"),
        (("complet",), "completed"),
        (("saving", "save result"), "saving"),
        (("validat",), "validating"),
        (("decoding", "decode"), "decoding"),
        (("generating", "diffusion", "step "), "generating"),
        (("encoding prompt",), "encoding_prompt"),
        (("conditioning",), "preparing_conditioning"),
        (("preparing fabric", "fabric appearance"), "preparing_fabric"),
        # Model-load substages must win over bare "initializ" (e.g. "Initializing FLUX").
        (
            (
                "reusing",
                "loading model",
                "loading flux",
                "flux ready",
                "download",
                "downloading",
                "transformer",
                "pipeline",
                "offload",
                "cache",
                "initializing flux",
                "in-memory",
            ),
            "loading_model",
        ),
        (("connect", "remote", "waiting for worker", "queued"), "initializing"),
    )
    for keys, stage in rules:
        if any(k in lower for k in keys):
            return stage
    return "processing"
