# Day 17 Resolution Benchmark (measured)

Hardware: NVIDIA RTX 3050 Laptop GPU 6GB · Model: FLUX.1-Kontext · Quantization: NF4 · Offload: `model_cpu_offload` · Attention: SDPA · Seed: 42 · Guidance: 2.5 · Steps fixed: 3 · Fabric: white/red floral (`30c9aaca…jpg`) · Match Fabric / `has_image=True`

All timings from `time.perf_counter()`. No fabricated values. Visual scores are human inspection of saved PNGs (not edge-energy alone).

| Resolution | Steps | Guidance | Runtime | Peak VRAM | Fabric Fidelity | Structure | Sharpness | Overall Quality | Status |
|------------|-------|----------|---------|-----------|-----------------|-----------|-----------|-----------------|--------|
| 384×384 | 3 | 2.5 | 553.0 s (~9.2 min) | 9476 MB | Good (white/red preserved) | OK silhouette; V-neck readable | Soft + mosaic/pixelation (edge=3.42 inflated by noise) | Fair — readable but not catalog | ok |
| 512×512 | 3 | 2.5 | 603.5 s (~10.1 min) | 9644 MB | Good | Clear top / short sleeve / neckline | Soft watercolor; better than 384 mosaic | **Best practical** among tested | ok |
| 640×640 | 3 | 2.5 | 1173.0 s (~19.5 min) | 9862 MB | Good | Similar structure | Still soft; no clear sharpness win | Reject default (2× time, soft) | ok |
| 768×768 | 3 | 2.5 | 1435.6 s (~23.9 min) | 10126 MB | Good (slightly washed) | Soft edges / white bleed | Softest by edge=1.81; still blurry | Reject default (slowest, still soft) | ok |

### Visual conclusion

Raising resolution to 640/768 **did not** produce product-photography sharpness on this stack. Softness is already present in the raw FLUX PNG. **512×512** is the best practical resolution: less mosaic than 384, similar softness to 640/768, ~half the runtime of 640.

Model init (once): **75.6 s**

Images: `outputs/generated_garments/day17_resolution_tests/resolution_{384,512,640,768}_steps3.png`

_All timings measured with `time.perf_counter()` on local RTX 3050 6GB. No fabricated values._
