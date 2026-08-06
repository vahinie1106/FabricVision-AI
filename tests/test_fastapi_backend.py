import os
import time
import uuid
import requests
import pytest
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:8000"
API_PREFIX = "/api/v1"
API_URL = f"{BASE_URL}{API_PREFIX}"

def create_sample_image(filename="test_garment.png", color=(100, 150, 200), size=(512, 512), fmt="PNG"):
    """Create a temporary PIL sample image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()

class TestFastAPIBackend:
    
    @classmethod
    def setup_class(cls):
        """Prepare test resources."""
        cls.valid_png = create_sample_image("sample_garment.png", color=(200, 100, 100), fmt="PNG")
        cls.valid_jpg = create_sample_image("sample_fabric.jpg", color=(100, 200, 100), fmt="JPEG")
        cls.person_img = create_sample_image("sample_person.png", color=(240, 200, 180), fmt="PNG")

    def test_01_health_check(self):
        """STEP 3.A: Health API Test."""
        start_time = time.time()
        resp = requests.get(f"{API_URL}/health")
        latency = round((time.time() - start_time) * 1000, 2)
        
        assert resp.status_code == 200, f"Expected status 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data.get("status") == "healthy", f"Unexpected status: {data}"
        assert "version" in data, "Missing version field in health response"
        print(f"[PASS] Health API Check (Latency: {latency} ms)")

    def test_02_semantic_analysis(self):
        """STEP 3.B: Semantic Analysis API Test."""
        start_time = time.time()
        files = {
            "garment_image": ("garment.png", self.valid_png, "image/png")
        }
        resp = requests.post(f"{API_URL}/analyze", files=files)
        latency = round(time.time() - start_time, 2)
        
        assert resp.status_code == 200, f"Semantic analysis failed ({resp.status_code}): {resp.text}"
        data = resp.json()
        assert data.get("status") == "completed", f"Status not completed: {data}"
        assert "metadata" in data, "Missing metadata in response"
        print(f"[PASS] Semantic Analysis API Check (Latency: {latency} s)")
        print(f"       Metadata: {data.get('metadata')}")

    def test_03_custom_generation_flow(self):
        """STEP 3.C: Custom Generation Job Creation & Polling Test."""
        files = {
            "fabric_image": ("fabric.jpg", self.valid_jpg, "image/jpeg")
        }
        form_data = {
            "garment_type": "kurti",
            "fit": "regular",
            "style": "casual"
        }
        
        # 1. Submit Job
        start_time = time.time()
        resp = requests.post(f"{API_URL}/generate", files=files, data=form_data)
        assert resp.status_code == 200, f"Generation submission failed ({resp.status_code}): {resp.text}"
        
        job_data = resp.json()
        job_id = job_data.get("job_id")
        assert job_id is not None, "No job_id returned"
        
        # Verify valid UUID
        val_uuid = uuid.UUID(job_id)
        assert str(val_uuid) == job_id, f"Invalid UUID format: {job_id}"
        print(f"[PASS] Generation Job Created: {job_id}")

        # 2. Poll Status
        completed = False
        status_data = None
        for attempt in range(15):
            status_resp = requests.get(f"{API_URL}/status/{job_id}")
            assert status_resp.status_code == 200, f"Status check failed: {status_resp.text}"
            status_data = status_resp.json()
            curr_status = status_data.get("status")
            print(f"       Polling job {job_id} -> status: {curr_status}, progress: {status_data.get('progress')}%")
            
            if curr_status in ["completed", "failed"]:
                completed = True
                break
            time.sleep(1)
            
        total_time = round(time.time() - start_time, 2)
        assert completed, f"Job did not complete in time. Final state: {status_data}"
        assert status_data.get("status") == "completed", f"Job failed: {status_data.get('error')}"
        assert status_data.get("result_url") is not None, "Missing result_url in completed job"
        print(f"[PASS] Generation Job Completed in {total_time} s. Result URL: {status_data.get('result_url')}")

    def test_04_virtual_tryon_flow(self):
        """STEP 3.D: Virtual Try-On Job Creation & Polling Test."""
        files = {
            "garment_image": ("garment.png", self.valid_png, "image/png"),
            "person_image": ("person.png", self.person_img, "image/png")
        }
        form_data = {
            "fit_preference": "regular",
            "background_action": "preserve"
        }
        
        # 1. Submit Job
        start_time = time.time()
        resp = requests.post(f"{API_URL}/tryon", files=files, data=form_data)
        assert resp.status_code == 200, f"Try-On submission failed ({resp.status_code}): {resp.text}"
        
        job_data = resp.json()
        job_id = job_data.get("job_id")
        assert job_id is not None, "No job_id returned"
        print(f"[PASS] Virtual Try-On Job Created: {job_id}")

        # 2. Poll Status
        completed = False
        status_data = None
        for attempt in range(15):
            status_resp = requests.get(f"{API_URL}/status/{job_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            curr_status = status_data.get("status")
            print(f"       Polling job {job_id} -> status: {curr_status}, progress: {status_data.get('progress')}%")
            
            if curr_status in ["completed", "failed"]:
                completed = True
                break
            time.sleep(1)
            
        total_time = round(time.time() - start_time, 2)
        assert completed, f"Job did not complete in time. Final state: {status_data}"
        assert status_data.get("status") == "completed", f"Job failed: {status_data.get('error')}"
        assert status_data.get("result_url") is not None, "Missing result_url in completed job"
        print(f"[PASS] Virtual Try-On Job Completed in {total_time} s. Result URL: {status_data.get('result_url')}")

    def test_05_storage_validation(self):
        """STEP 4: Storage & Format Validation Test."""
        # A) Test invalid file extension (.txt)
        txt_content = b"This is a text file, not an image."
        files_txt = {"garment_image": ("document.txt", txt_content, "text/plain")}
        resp_txt = requests.post(f"{API_URL}/analyze", files=files_txt)
        assert resp_txt.status_code == 400, f"Expected 400 for .txt file, got {resp_txt.status_code}"
        print("[PASS] Blocked .txt file upload correctly (400 Bad Request)")

        # B) Test corrupted image data
        corrupt_bytes = b"CORRUPTED_IMAGE_HEADER_1234567890"
        files_corrupt = {"garment_image": ("bad.png", corrupt_bytes, "image/png")}
        resp_corrupt = requests.post(f"{API_URL}/analyze", files=files_corrupt)
        assert resp_corrupt.status_code in [400, 500], f"Expected error code for corrupted image, got {resp_corrupt.status_code}"
        print(f"[PASS] Handled corrupted image file correctly ({resp_corrupt.status_code})")

        # C) Test valid PNG & JPEG extensions
        files_png = {"garment_image": ("valid.png", self.valid_png, "image/png")}
        resp_png = requests.post(f"{API_URL}/analyze", files=files_png)
        assert resp_png.status_code == 200, "PNG upload failed"
        
        files_jpg = {"garment_image": ("valid.jpg", self.valid_jpg, "image/jpeg")}
        resp_jpg = requests.post(f"{API_URL}/analyze", files=files_jpg)
        assert resp_jpg.status_code == 200, "JPG upload failed"
        print("[PASS] Accepted valid PNG and JPG formats successfully")

if __name__ == "__main__":
    test_suite = TestFastAPIBackend()
    test_suite.setup_class()
    
    print("==================================================")
    print("RUNNING FASTAPI BACKEND VALIDATION SUITE")
    print("==================================================")
    
    test_suite.test_01_health_check()
    test_suite.test_02_semantic_analysis()
    test_suite.test_03_custom_generation_flow()
    test_suite.test_04_virtual_tryon_flow()
    test_suite.test_05_storage_validation()
    
    print("==================================================")
    print("ALL BACKEND VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
