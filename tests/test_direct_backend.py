import os
import time
import uuid
import pytest
from PIL import Image
import io
from fastapi.testclient import TestClient

from backend_api.main import app

client = TestClient(app)

def create_sample_image(color=(100, 150, 200), size=(512, 512), fmt="PNG"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

def test_01_health_check():
    print("\n[STEP 1] Testing GET /api/v1/health...")
    start = time.time()
    resp = client.get("/api/v1/health")
    lat = round((time.time() - start) * 1000, 2)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    print(f" -> PASS: Health check status={data['status']}, version={data['version']} ({lat} ms)")

@pytest.mark.slow
def test_02_storage_validation():
    """
    Real Qwen2.5-VL semantic analysis through the live FastAPI app.

    Local weights for Qwen2.5-VL-3B-Instruct exist under models/, so
    /api/v1/analyze loads and runs the actual 3B vision-language model on the
    RTX 3050 6GB (real, potentially multi-minute GPU work), not a placeholder.
    Excluded from the default `pytest -q` run via the `slow` marker; run with
    `pytest -m slow` when you want a real hardware smoke test.
    """
    print("\n[STEP 4] Testing File Storage & Format Validation...")
    
    # Invalid format .txt
    txt_data = b"Hello text file"
    resp_txt = client.post("/api/v1/analyze", files={"garment_image": ("test.txt", txt_data, "text/plain")})
    assert resp_txt.status_code == 400
    print(" -> PASS: Blocked invalid extension .txt (400 Bad Request)")
    
    # Valid formats .png and .jpg
    png_data = create_sample_image(fmt="PNG")
    jpg_data = create_sample_image(fmt="JPEG")
    
    # We test analyzing png
    resp_png = client.post("/api/v1/analyze", files={"garment_image": ("valid.png", png_data, "image/png")})
    assert resp_png.status_code == 200
    print(" -> PASS: Accepted valid PNG image upload")

@pytest.mark.slow
def test_03_generation_flow():
    """
    Real end-to-end FLUX.1-Kontext generation through the live FastAPI app.

    generation_service.process_generation() forces allow_fallback=False, so this
    hits actual model weights (~10-13 min on RTX 3050 6GB) and, because
    BackgroundTasks run inside the ASGI call, TestClient blocks until it finishes.
    Excluded from the default `pytest -q` run via the `slow` marker; run with
    `pytest -m slow` when you want a real hardware smoke test.
    """
    print("\n[STEP 3.C] Testing POST /api/v1/generate & Status Polling...")
    jpg_data = create_sample_image(fmt="JPEG")
    resp = client.post("/api/v1/generate", files={"fabric_image": ("fabric.jpg", jpg_data, "image/jpeg")}, data={"garment_type": "kurti", "fit": "regular", "style": "casual"})
    assert resp.status_code == 200
    job_id = resp.json().get("job_id")
    assert job_id is not None
    print(f" -> PASS: Job created successfully with job_id={job_id}")
    
    status_resp = client.get(f"/api/v1/status/{job_id}")
    assert status_resp.status_code == 200
    print(f" -> PASS: Initial Job Status: {status_resp.json()}")

@pytest.mark.slow
def test_04_tryon_flow():
    """
    Real end-to-end CatVTON try-on through the live FastAPI app.

    Local CatVTON checkpoints exist under models/CatVTON, so /api/v1/tryon can
    load and run the actual diffusion pipeline instead of the composite
    fallback. Excluded from the default `pytest -q` run via the `slow` marker;
    run with `pytest -m slow` for a real hardware smoke test.
    """
    print("\n[STEP 3.D] Testing POST /api/v1/tryon & Status Polling...")
    g_data = create_sample_image(fmt="PNG")
    p_data = create_sample_image(fmt="PNG")
    resp = client.post("/api/v1/tryon", files={"garment_image": ("g.png", g_data, "image/png"), "person_image": ("p.png", p_data, "image/png")}, data={"fit_preference": "regular", "background_action": "preserve"})
    assert resp.status_code == 200
    job_id = resp.json().get("job_id")
    assert job_id is not None
    print(f" -> PASS: Virtual Try-On Job created successfully with job_id={job_id}")
    
    status_resp = client.get(f"/api/v1/status/{job_id}")
    assert status_resp.status_code == 200
    print(f" -> PASS: Initial Try-On Job Status: {status_resp.json()}")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING DIRECT FASTAPI BACKEND TEST SUITE")
    print("==================================================")
    test_01_health_check()
    test_02_storage_validation()
    test_03_generation_flow()
    test_04_tryon_flow()
    print("==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")
