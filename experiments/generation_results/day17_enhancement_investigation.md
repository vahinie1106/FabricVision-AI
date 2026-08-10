# Day 17 — Second-Stage Detail Enhancement Investigation

**Date:** 08 August 2026
**Hardware:** NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), 16 GB RAM, Windows, Python 3.11
**Baseline:** FLUX.1-Kontext, 4-bit NF4, `model_cpu_offload`, SDPA, fabric conditioning, Standard mode (512×512, 3 steps, guidance 2.5)

This report documents a real, measured investigation into whether a lightweight
super-resolution / detail-restoration model can be added as a second stage after
FLUX.1-Kontext to improve perceived sharpness **without** changing garment
identity or fabric fidelity. No model was integrated into the production
pipeline as a result of this investigation — see **Final Recommendation**.

All numbers below are labelled **Measured** (produced by
`scripts/benchmark_garment_enhancement.py` or a direct interactive check in
this session) or **Expected** (vendor/architecture claims not verified on
this machine). Nothing is fabricated.

---

## 1. Source image used for testing

Rather than reusing a synthetic/placeholder image, a real FLUX.1-Kontext
Standard-mode image was generated for this investigation
(`scripts/run_day17_fix_generation.py`) so the enhancement candidates would be
tested against genuine model output, not a stand-in.

- Source: `outputs/generated_garments/images/day17_enhancement_source.png`
- Metadata: `experiments/generation_results/day17_enhancement_source_exp.json`
- Garment: women's kurti, round neckline, three-quarter sleeve, slim fit
- Fabric: white base, red floral print, cotton, `color_source: fabric_pixels` (pixel-derived, not UI-forced)
- Resolution: 512×512
- **Measured** FLUX generation time: **829.44 s** (`pipeline_timings.total_pipeline_s` / `stats.generation_time_s`), 3 steps, per-step durations `[287.0s, 253.7s, 257.7s]`
- **Measured** peak VRAM during FLUX generation: **9643.8 MB** (`stats.peak_vram_mb`) — this exceeds the GPU's 6 GB physical VRAM, confirming the pipeline is spilling into Windows' CUDA shared-memory fallback under `model_cpu_offload`. This is consistent with — and helps explain — the heavy CPU↔GPU traffic and ~10–14 minute practical runtime already identified as the performance bottleneck.
- **Measured** prompt: 49/77 CLIP tokens, `truncated: false`

## 2. Candidate models investigated

| # | Candidate | Architecture | File size | Loader | Why considered |
|---|-----------|--------------|-----------|--------|-----------------|
| 1 | `realesr-general-x4v3.pth` | SRVGGNetCompact (Real-ESRGAN "general" compact variant) | ~1.2 MB (compact) | `spandrel` | Smallest/fastest Real-ESRGAN family model; designed for general real-world restoration; no `basicsr`/`facexlib` dependency via spandrel |
| 2 | `RealESRGAN_x2plus.pth` | RRDBNet (original Real-ESRGAN, x2 variant) | ~64 MB | `spandrel` | Larger, more established Real-ESRGAN architecture; tested to see if a heavier network avoids the artifact seen in candidate 1 |

`spandrel` was chosen as the loader for both because it is a pure-PyTorch,
Windows-friendly library that loads common SR architectures (SRVGGNetCompact,
RRDBNet, and others) directly from a `.pth` checkpoint without requiring the
unmaintained `basicsr`/`facexlib`/`gfpgan` dependency chain that the official
Real-ESRGAN repo pulls in (these have known install friction on Windows +
Python 3.11).

Both models run entirely locally, fully offline, no external API calls.

## 3. Compatibility (Measured)

| Item | Candidate 1 (x4v3 compact) | Candidate 2 (x2plus RRDBNet) |
|---|---|---|
| Windows install | pip install spandrel only | same |
| PyTorch/CUDA compatibility | works with existing torch 2.6.0+cu124 env | same |
| Input resolution | 512×512 (arbitrary) | 512×512 (arbitrary) |
| Native network output | 2048×2048 (4×) | 1024×1024 (2×) |
| Final output (requested outscale=2×, bicubic resize down from native) | 1024×1024 | 1024×1024 |
| Runs alongside FLUX in same process without OOM | Yes (tested after FLUX pipeline was idle) | Yes |

## 4. Performance (Measured)

Benchmarks produced by `scripts/benchmark_garment_enhancement.py`, raw JSON in
`experiments/generation_results/`:

| Run | Model | Tile | Inference time | Total enhancement time | Peak VRAM |
|---|---|---|---|---|---|
| `day17_enhancement_v1` | x4v3 compact | 256 | 0.709 s | 0.979 s | 123.5 MB |
| `day17_enhancement_v2_notile` | x4v3 compact | none (10000) | 0.362 s | 0.552 s | 199.8 MB |
| `day17_enhancement_v3_rrdb` | x2plus RRDBNet | 256 | 1.080 s | 1.651 s | 300.0 MB |

CPU RAM overhead: +300–415 MB for either candidate (well within the 16 GB budget).

**Runtime conclusion:** Both candidates are effectively free relative to FLUX.
Adding either to the pipeline would move total runtime from **829.44 s → ~830–831 s**,
an increase of well under 1%. **Runtime is not the blocking factor for either candidate.**

## 5. Visual / fidelity result (Measured — this is the actual finding)

Both candidates were run on the *same* real FLUX output and visually compared
against it and against the known floral fabric ground truth (white base, red
floral elements). Result for **both** architectures, independently:

- A regular, high-contrast **basket-weave / grid texture is hallucinated across the entire garment**, including flat white regions of the source image that contain no such texture.
- The hallucinated weave is present **with tiling and without tiling** (`v1` vs `v2_notile`), which rules out tiling-seam artifacts as the cause — it is a property of the network's learned prior, not an implementation bug in the benchmark script.
- The hallucinated weave is present in **both a compact SRVGGNet model and a full RRDBNet model** — two different, independently-trained architectures — which rules out a single bad checkpoint and points to a systematic cause: these Real-ESRGAN-family models are trained on real macro-photography of textiles (with real, high-frequency, regular weave structure) as part of their restoration training data, so they have learned a strong prior that "in-focus fabric has visible thread/weave structure" and inject it into any soft, low-texture fabric region.
- The soft floral print is preserved in terms of gross color/placement (no color-clone or hue shift, no relocation of the red floral motifs), so the model is not destroying the *garment structure or silhouette*. The problem is specifically a **fabric-texture hallucination**, which directly violates the Part 5/Part 10 requirement: "must NOT invent... textures... that were not present in the original."

**Mitigation attempted (Measured, informal):** To check whether the artifact
could be controlled with strength, the `v1` enhanced output was alpha-blended
with the (bicubic-resized) original at 30% and 50% enhancement strength
(`outputs/generated_garments/enhanced/day17_enhancement_blend_30.png`,
`..._blend_50.png`). The hallucinated weave pattern remained clearly visible
at both blend strengths because it is high-frequency, high-contrast content
that survives linear blending far more than the underlying soft color does.
Simple output blending is **not** an adequate fix.

## 6. Fabric fidelity test (Part 13)

Ground truth uploaded fabric: white/light base, red floral elements (per
`fabric_metadata` in the source generation: `dominant_colors: [white, red]`,
`pattern: floral`, `color_source: fabric_pixels`).

| Check | Result |
|---|---|
| Base color preserved (white/cream) | Pass — preserved in both candidates |
| Red floral motif color preserved | Pass — no hue shift |
| Floral motif position/shape preserved | Pass — no smearing into a different pattern |
| No new pattern invented | **FAIL** — a weave/grid texture not present in the source or the uploaded fabric is added on top of the floral print by both candidates |
| Garment silhouette / neckline / sleeve preserved | Pass — SR models operate pixel-locally; they do not redraw structure |

**Verdict: fabric fidelity check fails** for both tested candidates because of
the hallucinated weave texture, even though color and garment structure are
preserved.

## 7. Garment structure result (Part 10/11 checklist)

| Item | Result |
|---|---|
| Silhouette preservation | Pass |
| Neckline preservation | Pass |
| Sleeve preservation | Pass |
| Fabric color preservation | Pass |
| Fabric pattern preservation (position/shape of floral motifs) | Pass |
| Fabric texture preservation | **FAIL** — hallucinated weave added |
| No invented buttons/seams/prints/logos/embroidery | Pass — nothing structural was invented, only surface texture |
| Edge/fine-detail clarity improvement | Ambiguous — technically sharper at the pixel level, but the added sharpness is the hallucinated texture, not genuine recovered garment detail |

## 8. Installation requirements (for completeness)

```
pip install spandrel
```
Model checkpoints (`.pth`) downloaded manually to `models/enhancement/`. No
`basicsr`, `facexlib`, or `gfpgan` required (these were deliberately avoided
due to known Windows/Python 3.11 install friction and unmaintained status).

## 9. Recommendation

**OPTION C** (per task Part 17): No suitable enhancement model, among those
realistically installable on this RTX 3050 6 GB / Windows / Python 3.11
environment, currently provides a reliable improvement that satisfies the
fidelity requirement.

- Both tested candidates (SRVGGNetCompact `x4v3` and RRDBNet `x2plus`) are
  effectively **free** in time/VRAM/RAM terms (≤1.7 s, ≤300 MB VRAM), so
  **performance is not the obstacle**.
- Both tested candidates **hallucinate a fabric weave texture that is not
  present in the source image**, which is an explicit, hard failure condition
  per Parts 5, 10, and 13 of the task. This was verified across two
  independent architectures, with and without tiling, and did not go away
  under partial-strength blending — i.e. it is a real, repeatable, measured
  finding, not a one-off glitch.
- Forcing either model into production would violate the "no hallucinated
  detail" requirement, so **neither is integrated**.

**Do not force an enhancement model into the default pipeline.** Per Option C,
this is documented as deferred rather than shipped:

> High-quality garment detail enhancement should be revisited either (a) with
> a model fine-tuned or selected specifically for soft, low-texture diffusion
> output rather than real-world camera restoration (e.g. a lightweight
> diffusion-based refiner, or a Real-ESRGAN variant fine-tuned on FLUX/SD
> outputs), or (b) on a stronger GPU/server where more capable
> restoration/upscaling models (and the iteration time to evaluate them
> properly) become practical. On the current RTX 3050 6 GB laptop, the two
> realistic lightweight candidates available today do not meet the fidelity
> bar.

No `GARMENT_ENHANCEMENT_ENABLED` flag or pipeline integration was added, since
there is nothing validated to gate behind it. `scripts/benchmark_garment_enhancement.py`
remains in the repo as a reusable, isolated test harness for evaluating future
candidates without touching the production pipeline.

## 10. Artifacts produced by this investigation

- `scripts/benchmark_garment_enhancement.py` — isolated benchmark harness (spandrel loader, tiled upscale, VRAM/RAM/timing measurement, color-fidelity signal)
- `models/enhancement/realesr-general-x4v3.pth`, `models/enhancement/RealESRGAN_x2plus.pth` — downloaded checkpoints
- `experiments/generation_results/day17_enhancement_source_exp.json` — FLUX source generation metadata
- `experiments/generation_results/day17_enhancement_v1_bench.json`, `..._v2_notile_bench.json`, `..._v3_rrdb_bench.json` — measured benchmark reports
- `outputs/generated_garments/images/day17_enhancement_source.png` — real FLUX output used as ground truth
- `outputs/generated_garments/enhanced/day17_enhancement_v1_enhanced.png`, `..._v2_notile_enhanced.png`, `..._v3_rrdb_enhanced.png` — enhanced outputs showing the hallucinated weave artifact
- `outputs/generated_garments/enhanced/day17_enhancement_blend_30.png`, `..._blend_50.png` — blend-strength mitigation check (informal, not a full benchmark run)
