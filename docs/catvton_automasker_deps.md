# AutoMasker / detectron2 dependency notes (CatVTON)

Do **not** blindly install random detectron2 wheels. Prefer the versions
documented by the vendored CatVTON checkout.

## Source of truth

- `models/CatVTON/requirements.txt`
- Vendored tree: `models/CatVTON/detectron2/` (present in this repo)
- Weights expected under: `models/CatVTON/{model,SCHP,densepose|DensePose}`

## Current project environment (measured)

- Python 3.11.x
- PyTorch 2.6.0+cu124
- CatVTON pins `torch==2.4.0` — **do not downgrade** the shared FabricVision
  venv torch just to satisfy that pin. Prefer installing AutoMasker *Python*
  deps into the existing venv, or create a dedicated try-on env if needed.

## Minimum packages that currently block AutoMasker

On this machine, `import detectron2` (even from the vendored tree) fails with:

`ModuleNotFoundError: No module named 'fvcore'`

Install only the CatVTON-documented stack first:

```bash
# from project venv
pip install "fvcore==0.1.5.post20221221" cloudpickle omegaconf iopath "pycocotools==2.0.8"
```

Vendored `detectron2` is imported by placing `models/CatVTON` on `sys.path`
(as `person_masker.try_automasker_mask` already does). It is **not** installed
into site-packages.

Then re-probe:

```bash
python scripts/tryon/run_catvton_quality_matrix.py --plan-only
```

`automasker_ready` must be true before running cell `B_automasker_30`.

## Runtime flags

- `CATVTON_MASK_STRATEGY=auto|automasker|grabcut|provided_only`
- `CATVTON_USE_AUTOMASKER=true` (forced by matrix cell B/C)
- `CATVTON_REQUIRE_REAL=true` (quality scripts set this)

GrabCut remains the supported fallback when AutoMasker is unavailable.
LOW_VRAM / infrastructure paths must keep working without detectron2.
