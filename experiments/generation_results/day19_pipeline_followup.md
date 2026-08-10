# Day-19 FLUX / CatVTON Pipeline Follow-up Benchmark

**Date:** 09 August 2026  
**Hardware:** NVIDIA RTX 3050 Laptop GPU — **6GB physical VRAM**, 16GB RAM, Windows/WDDM

## Terminology

| Term | Meaning |
|------|---------|
| Physical VRAM | 6GB on-device GDDR |
| PyTorch allocated / reserved | Process CUDA memory stats |
| Windows/WDDM shared GPU memory | Host-backed accounting (often ~9.5–10GB in prior logs) — **not** physical VRAM |

## FLUX configuration (unchanged mode targets)

| Mode | Resolution | Steps | Guidance | Preencode | max_seq |
|------|----------:|------:|---------:|:---------:|--------:|
| Preview | 384 | 3 | 2.5 | true | 128 |
| Standard | 512 | 3 | 2.5 | true | 128 |
| Production | 512 | 4 | 2.5 | true | 128 |

## Measured wall-clock table

| Test | Resolution | Steps | T5 | Diffusion | VAE | Total | PyTorch VRAM | Quality |
|------|----------:|------:|---:|----------:|----:|------:|-------------:|---------|
| Baseline (Day-17 Standard cold) | 512 | 3 | ~48–50s @ seq=128 (preencode path) | dominates (~120–400s/step class) | small vs denoise | **~603s cold** | WDDM shared acct ~9.5–10GB; physical 6GB | Soft usable |
| Day-19 residency opt (encode→CPU embeds + encoder eviction + alloc conf) | 512 | 3 | Not measured (full cold rerun) | Not measured | Not measured | **Not measured** | Not measured | Prompt/construction wording improved; no ESRGAN |
| Quality candidate 4-step Production | 512 | 4 | Not measured | Not measured | Not measured | Day-17 class **~12–20 min** | Same class | Slightly more denoise |
| Quality candidate 5-step | 512 | 5 | — | — | — | **Not measured** (rejected without evidence; would multiply offload cost) | — | — |
| 640×640 probe (Day-17) | 640 | 3 | — | — | — | **~19.5 min** | ~10GB shared acct | Still soft — rejected |

## Day-19 code changes affecting runtime (not yet re-timed end-to-end)

1. Keep prompt embeds on **CPU** after encode (avoid double GPU residency before denoise).
2. Explicit `_evict_text_encoders` + `maybe_free_model_hooks` after T5/CLIP encode.
3. Default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when unset (fragmentation).
4. Prompt layers emphasize construction (neckline/seams/hem) within CLIP budget.
5. Garment validator adds diagnostic `sharpness_edge_var` (does not alone fail soft images).

**Honest statement:** A fresh Standard cold wall-clock was **not** completed in this session (~10 min GPU job). Do not claim a new total runtime without that measurement.

## CatVTON mask path

| Path | Default? | Status |
|------|:--------:|--------|
| Provided mask | if uploaded | supported |
| AutoMasker (DensePose+SCHP) | `CATVTON_USE_AUTOMASKER=true` | attempted; requires importable detectron2 — **not available in current env** → falls back |
| OpenCV GrabCut | **default on** (`CATVTON_USE_GRABCUT=true`) | implemented; no new deps |
| Box fallback | last resort | labeled `mask_source=box_fallback` |

Blend preview sets `was_fallback_used=true`, `status=completed_with_fallback`, and frontend toast when metadata present. `CATVTON_REQUIRE_REAL=true` disables silent blend.

## Job lifecycle

Day-18 fix retained: no soft auto-advance for garment/try-on; backend `stage` authoritative. Not re-broken.
