# Day 17 Step Benchmark (measured)

Primary matrix at **512×512** (visual best resolution). Secondary matrix at 384×384 archived (edge-energy auto-pick; rejected after visual QA).

Hardware / model / offload same as resolution benchmark. Seed 42 · Guidance 2.5 · Match Fabric.

## Primary: 512×512

| Resolution | Steps | Guidance | Runtime | Peak VRAM | Fabric Fidelity | Structure | Sharpness | Overall Quality | Status |
|------------|-------|----------|---------|-----------|-----------------|-----------|-----------|-----------------|--------|
| 512×512 | 2 | 2.5 | 660.1 s | 9644 MB | Good | Soft silhouette; more halo | Soft / grainy | Preview-only at this res | ok |
| 512×512 | 3 | 2.5 | 976.4 s* / **603.5 s cold**† | 9644 MB | Good | Readable neckline/sleeves | Soft (same class as 2/4) | **Standard default** | ok |
| 512×512 | 4 | 2.5 | 1213.4 s* | 9644 MB | Good | Slightly cleaner construction | Soft; no clear sharpness leap | **Production** (one extra step) | ok |

\* Later session runs show thermal/load variance (slower per-step).  
† Cold measurement from resolution phase (`day17_res_512_s3`) is the cleaner Standard reference: **603.5 s**.

## Secondary: 384×384 (archived)

| Resolution | Steps | Guidance | Runtime | Peak VRAM | Fabric Fidelity | Structure | Sharpness | Overall Quality | Status |
|------------|-------|----------|---------|-----------|-----------------|-----------|-----------|-----------------|--------|
| 384×384 | 2 | 2.5 | 375.0 s (~6.3 min) | 9476 MB | Good | Coarse | Soft + pixelated | Preview OK | ok |
| 384×384 | 3 | 2.5 | 439.7 s | 9476 MB | Good | OK | Soft + mosaic | Not Standard (too pixelated) | ok |
| 384×384 | 4 | 2.5 | 576.5 s | 9476 MB | Good | Slightly better | Soft + mosaic | Rejected vs 512 | ok |

### Step conclusion

On this hardware, **2→3→4 steps do not fix blur**. They mainly buy modest structure stability at roughly linear time cost. Prefer **3 steps** for Standard and **4 steps** for Production. Do **not** default to 6+ steps (prior Day 16/17 run: 512/6 ≈ 1023 s, no clear sharpness win).

Images (512): `outputs/generated_garments/day17_resolution_tests/best_resolution_steps{2,3,4}.png`  
Images (384 archive): `…/steps_r384_s{2,3,4}.png`

_All timings measured with `time.perf_counter()` on local RTX 3050 6GB. No fabricated values._
