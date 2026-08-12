#!/usr/bin/env python3
"""Kaggle end-to-end integration (run TWICE from a clean startup).

Does NOT redesign the UI. Proves real inference — HTTP 200 alone is not enough:

  Browser UI :3000 (/proxy/3000/)
  -> API :8000 (/proxy/8000/api/v1)
  -> job manager
  -> real FLUX Production 768x768 / 12 / 3.0 on cuda:0
  -> PNG on disk + /outputs/... media reachable
  -> real CatVTON on cuda:1
  -> PNG on disk + /outputs/... media reachable

Usage (inside Kaggle after ``python scripts/run_kaggle.py`` is READY):

    python scripts/kaggle_e2e_integration.py --run 1
    # stop services, restart cleanly, then:
    python scripts/kaggle_e2e_integration.py --run 2

Options:
    --skip-inference   routes + health + UI markers only (no GPU work)
    --api URL          default http://127.0.0.1:8000/api/v1  (notebook-local)
    --frontend URL     default http://127.0.0.1:3000
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "kaggle" / "e2e_integration"

ROUTES = [
    "/",
    "/about",
    "/projects",
    "/studio",
    "/studio/custom-garment",
    "/studio/virtual-tryon",
    "/studio/semantic-analysis",
]

HOME_MARKERS = (
    "FabricVision-AI 2.0",
    "Transform Fabrics Into",
    "Start Creating",
    "Explore Technology",
    "Input Concept",
    "Raw Linen Fabric",
    "FLUX Synthesis Active",
    "Generated AI Garment",
    "AI Fashion Intelligence",
    "Virtual Try-On",
    "Semantic Extraction",
)


def http_status(url: str, timeout: float = 15.0) -> int:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return 0


def http_get_bytes(url: str, timeout: float = 60.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        return int(exc.code), body
    except Exception:
        return 0, b""


def get_json(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = "----fv_e2e"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")
    for name, path in files.items():
        data = path.read_bytes()
        ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def post_form(url: str, fields: dict[str, str], files: dict[str, Path]) -> dict:
    body, boundary = multipart(fields, files)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def poll(api: str, job_id: str, timeout_s: int = 3600) -> dict:
    t0 = time.time()
    last: dict = {}
    while True:
        last = get_json(f"{api}/status/{job_id}")
        elapsed = int(time.time() - t0)
        print(
            f"  [{elapsed}s] {last.get('status')} {last.get('stage')} "
            f"{last.get('progress')} {last.get('current_step')}",
            flush=True,
        )
        if last.get("status") in ("completed", "failed"):
            return last
        if elapsed > timeout_s:
            last["status"] = "failed"
            last["error"] = f"timeout after {timeout_s}s"
            return last
        time.sleep(10)


def pick_fabric() -> Path:
    preferred = [
        ROOT / "tests" / "test_images" / "test_img_1.jpg",
        ROOT / "data" / "uploads" / "30c9aacaacf444169b3959f2171b4942.jpg",
        ROOT / "data" / "uploads" / "val_cotton.png",
    ]
    for p in preferred:
        if p.is_file():
            return p
    raise FileNotFoundError("No fabric image — pass --fabric")


def pick_person() -> Path:
    for p in [
        ROOT / "tests" / "test_images" / "test_img_2.jpg",
        ROOT / "tests" / "test_images" / "test_img_1.jpg",
    ]:
        if p.is_file():
            return p
    raise FileNotFoundError("No person image")


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def normalize_result_url(result_url: str | None) -> str | None:
    if not result_url:
        return None
    # Strip accidental absolute loopback (must never be browser-facing).
    for prefix in (
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://127.0.0.1:8000",
        "https://localhost:8000",
    ):
        if result_url.startswith(prefix):
            return result_url[len(prefix) :] or "/"
    return result_url


def browser_media_url(result_url: str | None) -> str | None:
    """Browser-facing path under split Kaggle proxy (UI on :3000)."""
    rel = normalize_result_url(result_url)
    if not rel:
        return None
    if rel.startswith("/proxy/"):
        return rel
    if not rel.startswith("/"):
        rel = f"/{rel}"
    return f"/proxy/8000{rel}"


def resolve_output(api_origin: str, result_url: str | None) -> Path | None:
    rel = normalize_result_url(result_url)
    if not rel:
        return None
    # Prefer local OUTPUT_DIR mapping for /outputs/...
    if "/outputs/" in rel:
        tail = rel.split("/outputs/", 1)[1]
        candidates = [
            ROOT / "outputs" / tail,
            ROOT / "data" / "outputs" / tail,
            Path("outputs") / tail,
        ]
        for c in candidates:
            if c.is_file():
                return c
    # Download via notebook-local API origin (127.0.0.1:8000 inside Kaggle).
    try:
        url = rel
        if rel.startswith("/"):
            url = api_origin.rstrip("/") + rel
        dest = OUT / f"dl_{Path(rel).name}"
        urllib.request.urlretrieve(url, dest)
        return dest if dest.is_file() and dest.stat().st_size > 0 else None
    except Exception:
        return None


def media_http_ok(api_origin: str, result_url: str | None) -> dict:
    rel = normalize_result_url(result_url)
    if not rel:
        return {"ok": False, "status": 0, "bytes": 0, "url": None}
    url = rel if rel.startswith("http") else api_origin.rstrip("/") + rel
    status, body = http_get_bytes(url)
    png = body[:8] == b"\x89PNG\r\n\x1a\n" or body[:3] == b"\xff\xd8\xff"
    return {
        "ok": status == 200 and len(body) > 100 and png,
        "status": status,
        "bytes": len(body),
        "url": url,
        "browser_url": browser_media_url(rel),
        "is_loopback_browser_url": bool(
            rel.startswith("http://127.0.0.1") or rel.startswith("http://localhost")
        ),
    }


def device_looks_like(device: object, want: str) -> bool:
    s = str(device or "").strip().lower()
    if not s:
        return False
    if s == want:
        return True
    # Accept "0" / "cuda:0" variants.
    if want.startswith("cuda:") and s in (want, want.split(":")[-1], f"cuda:{want.split(':')[-1]}"):
        return True
    return want in s


def run_flux(api: str, fabric: Path) -> dict:
    print("\n=== REAL FLUX Production 768x12 ===", flush=True)
    t0 = time.time()
    init = post_form(
        f"{api}/generate",
        {
            "garment_type": "dress",
            "fit": "Slim Fit",
            "style": "Casual",
            "gender": "Women",
            "season": "Summer",
            "occasion": "Casual",
            "fabric": "Cotton",
            "material": "Cotton",
            "texture": "Smooth",
            "color": "match_fabric",
            "sleeve": "Short",
            "neckline": "Round",
            "generation_mode": "Production",
        },
        {"fabric_image": fabric},
    )
    job_id = init["job_id"]
    st = poll(api, job_id)
    meta = st.get("metadata") or {}
    api_origin = api.replace("/api/v1", "")
    result_url = normalize_result_url(st.get("result_url"))
    out_path = resolve_output(api_origin, result_url)
    dims = image_size(out_path) if out_path else None
    media = media_http_ok(api_origin, result_url)
    height = meta.get("height") or meta.get("output_height")
    width = meta.get("width") or meta.get("output_width")
    steps = meta.get("num_inference_steps") or meta.get("steps")
    guidance = meta.get("guidance_scale")
    gpu = meta.get("device") or meta.get("gpu") or meta.get("flux_device")
    demoted = False
    if dims and dims != (768, 768):
        demoted = True
    if height and int(height) != 768:
        demoted = True
    if width and int(width) != 768:
        demoted = True

    ok = (
        st.get("status") == "completed"
        and bool(meta.get("was_real_flux_used"))
        and not bool(meta.get("was_fallback_used"))
        and dims == (768, 768)
        and int(steps or 0) == 12
        and abs(float(guidance or 0) - 3.0) < 1e-6
        and not demoted
        and bool(out_path)
        and bool(media.get("ok"))
        and not bool(media.get("is_loopback_browser_url"))
        and device_looks_like(gpu, "cuda:0")
    )
    return {
        "ok": ok,
        "job_id": job_id,
        "http_status": 200 if st.get("status") == "completed" else 500,
        "status": st.get("status"),
        "error": st.get("error"),
        "was_real_flux_used": meta.get("was_real_flux_used"),
        "was_fallback_used": meta.get("was_fallback_used"),
        "requested_resolution": "768x768",
        "actual_output_dims": f"{dims[0]}x{dims[1]}" if dims else None,
        "steps": steps,
        "guidance": guidance,
        "height": height,
        "width": width,
        "generation_time_s": meta.get("generation_time_s"),
        "model_load_time_s": meta.get("model_load_time_s") or meta.get("load_time_s"),
        "peak_vram_mb": meta.get("peak_vram_mb"),
        "result_url": result_url,
        "browser_media_url": media.get("browser_url"),
        "media_http": media,
        "output_path": str(out_path) if out_path else None,
        "wall_clock_s": round(time.time() - t0, 2),
        "gpu": gpu,
        "demoted_resolution": demoted,
        "oom": "out of memory" in str(st.get("error") or "").lower(),
    }


def run_tryon(api: str, person: Path, garment: Path) -> dict:
    print("\n=== REAL CatVTON try-on ===", flush=True)
    t0 = time.time()
    init = post_form(
        f"{api}/tryon",
        {
            "fit_preference": "regular",
            "background_action": "keep",
            "garment_type": "dress",
        },
        {"person_image": person, "garment_image": garment},
    )
    job_id = init["job_id"]
    st = poll(api, job_id)
    meta = st.get("metadata") or {}
    api_origin = api.replace("/api/v1", "")
    result_url = normalize_result_url(st.get("result_url"))
    out_path = resolve_output(api_origin, result_url)
    dims = image_size(out_path) if out_path else None
    media = media_http_ok(api_origin, result_url)
    gpu = meta.get("device") or meta.get("gpu") or meta.get("catvton_device")
    ok = (
        st.get("status") == "completed"
        and bool(meta.get("was_real_catvton_used"))
        and not bool(meta.get("was_fallback_used"))
        and bool(out_path)
        and bool(media.get("ok"))
        and not bool(media.get("is_loopback_browser_url"))
        and device_looks_like(gpu, "cuda:1")
    )
    return {
        "ok": ok,
        "job_id": job_id,
        "http_status": 200 if st.get("status") == "completed" else 500,
        "status": st.get("status"),
        "error": st.get("error"),
        "was_real_catvton_used": meta.get("was_real_catvton_used"),
        "was_fallback_used": meta.get("was_fallback_used"),
        "mask_source": meta.get("mask_source"),
        "inference_time_s": meta.get("inference_time_s"),
        "output_dims": f"{dims[0]}x{dims[1]}" if dims else None,
        "result_url": result_url,
        "browser_media_url": media.get("browser_url"),
        "media_http": media,
        "output_path": str(out_path) if out_path else None,
        "wall_clock_s": round(time.time() - t0, 2),
        "gpu": gpu,
    }


def check_home_ui(frontend: str) -> dict:
    status, body = http_get_bytes(f"{frontend.rstrip('/')}/")
    text = body.decode("utf-8", errors="replace")
    missing = [m for m in HOME_MARKERS if m not in text]
    has_loopback_api = ("http://127.0.0.1:8000" in text) or ("http://localhost:8000" in text)
    return {
        "ok": status == 200 and not missing and not has_loopback_api,
        "status": status,
        "missing_markers": missing,
        "has_loopback_api_in_html": has_loopback_api,
    }


def check_client_bundle() -> dict:
    """Static check against frontend/.next when present (built by run_kaggle)."""
    next_dir = ROOT / "frontend" / ".next"
    if not next_dir.exists():
        return {"ok": None, "reason": "no_frontend_.next"}
    needles = (
        b"http://127.0.0.1:8000",
        b"http://localhost:8000",
    )
    proxy_hit = False
    loop_hits: list[str] = []
    for root in (next_dir / "static",):
        if not root.exists():
            continue
        for path in root.rglob("*.js"):
            try:
                data = path.read_bytes()
            except Exception:
                continue
            if b"/proxy/8000" in data:
                proxy_hit = True
            for n in needles:
                if n in data:
                    loop_hits.append(str(path.relative_to(ROOT)))
                    break
            if len(loop_hits) >= 3:
                break
    return {
        "ok": proxy_hit and not loop_hits,
        "has_proxy_8000": proxy_hit,
        "loopback_hits": loop_hits,
    }


def gpu_role_snapshot() -> dict:
    try:
        import torch
        from src.common.models.device_manager import DeviceManager

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "flux": DeviceManager.resolve_role_device("flux", "cuda:0"),
            "catvton": DeviceManager.resolve_role_device("catvton", "cuda:1"),
            "dual": DeviceManager.dual_gpu_residency_enabled(),
            "FLUX_CUDA_DEVICE": os.environ.get("FLUX_CUDA_DEVICE"),
            "CATVTON_CUDA_DEVICE": os.environ.get("CATVTON_CUDA_DEVICE"),
            "FABRICVISION_DUAL_GPU": os.environ.get("FABRICVISION_DUAL_GPU"),
            "FLUX_PRODUCTION_RESOLUTION": os.environ.get("FLUX_PRODUCTION_RESOLUTION"),
            "FLUX_PRODUCTION_STEPS": os.environ.get("FLUX_PRODUCTION_STEPS"),
            "FLUX_PRODUCTION_GUIDANCE": os.environ.get("FLUX_PRODUCTION_GUIDANCE"),
            "FLUX_PRODUCTION_NO_OOM_FALLBACK": os.environ.get(
                "FLUX_PRODUCTION_NO_OOM_FALLBACK"
            ),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle FE<->API E2E integration")
    parser.add_argument("--run", type=int, default=1, help="Run number (1 or 2)")
    parser.add_argument("--api", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--frontend", default="http://127.0.0.1:3000")
    parser.add_argument("--fabric", default="")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "run": args.run,
        "frontend_url": args.frontend,
        "backend_url": args.api,
        "browser_api_url_expected": "/proxy/8000/api/v1",
        "routes": {},
        "api_health": {},
        "home_ui": {},
        "client_bundle": check_client_bundle(),
        "gpu_roles": gpu_role_snapshot(),
    }

    print(f"=== E2E RUN #{args.run} ===", flush=True)
    for route in ROUTES:
        code = http_status(f"{args.frontend.rstrip('/')}{route}")
        report["routes"][route] = code
        print(f"  FE {route} -> {code}", flush=True)

    report["home_ui"] = check_home_ui(args.frontend)
    print(f"  Home UI -> {report['home_ui']}", flush=True)

    report["api_health"]["/api/v1/health"] = http_status(
        f"{args.api.rstrip('/')}/health"
    )
    report["api_health"]["/docs"] = http_status(
        args.api.replace("/api/v1", "") + "/docs"
    )
    print(f"  API health -> {report['api_health']}", flush=True)
    print(f"  Client bundle -> {report['client_bundle']}", flush=True)
    print(f"  GPU roles -> {report['gpu_roles']}", flush=True)

    if not args.skip_inference:
        fabric = Path(args.fabric) if args.fabric else pick_fabric()
        person = pick_person()
        report["fabric"] = str(fabric)
        report["person"] = str(person)
        report["flux"] = run_flux(args.api.rstrip("/"), fabric)
        print(json.dumps(report["flux"], indent=2)[:2500], flush=True)
        if report["flux"].get("demoted_resolution") or report["flux"].get("oom"):
            print(
                "FLUX FAILED CLEARLY (OOM or demoted resolution) — not claiming success",
                flush=True,
            )
        garment = fabric
        flux_out = report["flux"].get("output_path")
        if flux_out and Path(flux_out).is_file():
            garment = Path(flux_out)
        report["tryon"] = run_tryon(args.api.rstrip("/"), person, garment)
        print(json.dumps(report["tryon"], indent=2)[:2500], flush=True)

    out_path = OUT / f"run_{args.run}_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWROTE {out_path}", flush=True)

    route_ok = all(
        report["routes"].get(r) in (200, 307, 308)
        for r in ("/", "/studio/custom-garment", "/studio/virtual-tryon")
    )
    health_ok = report["api_health"].get("/api/v1/health") == 200
    docs_ok = report["api_health"].get("/docs") == 200
    ui_ok = bool(report.get("home_ui", {}).get("ok"))
    bundle = report.get("client_bundle") or {}
    bundle_ok = bundle.get("ok") in (True, None)  # None = no build on disk yet
    flux_ok = bool(report.get("flux", {}).get("ok")) if not args.skip_inference else True
    tryon_ok = bool(report.get("tryon", {}).get("ok")) if not args.skip_inference else True
    ok = route_ok and health_ok and docs_ok and ui_ok and bundle_ok and flux_ok and tryon_ok
    print(f"RUN #{args.run} ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
