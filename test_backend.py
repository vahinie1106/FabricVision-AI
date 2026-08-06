import os
import time
import requests

API_URL = "http://127.0.0.1:8000/api/v1"

# Create a dummy image for testing
test_image_path = "test_image.png"
from PIL import Image
img = Image.new('RGB', (512, 512), color = 'red')
img.save(test_image_path)

def test_health():
    print("Testing /health...")
    resp = requests.get(f"{API_URL}/health")
    print(resp.json())

def test_semantic_analysis():
    print("\nTesting /analyze...")
    with open(test_image_path, "rb") as f:
        files = {"garment_image": f}
        resp = requests.post(f"{API_URL}/analyze", files=files)
    print(resp.json())

def test_generation():
    print("\nTesting /generate...")
    with open(test_image_path, "rb") as f:
        files = {"fabric_image": f}
        data = {
            "garment_type": "dress",
            "fit": "regular",
            "style": "casual"
        }
        resp = requests.post(f"{API_URL}/generate", files=files, data=data)
    
    res_data = resp.json()
    print(res_data)
    
    if "job_id" in res_data:
        job_id = res_data["job_id"]
        print(f"Polling job status for {job_id}...")
        for _ in range(5):
            status_resp = requests.get(f"{API_URL}/status/{job_id}")
            print(status_resp.json())
            if status_resp.json().get("status") in ["completed", "failed"]:
                break
            time.sleep(2)

def test_tryon():
    print("\nTesting /tryon...")
    with open(test_image_path, "rb") as f1, open(test_image_path, "rb") as f2:
        files = {
            "garment_image": f1,
            "person_image": f2
        }
        data = {
            "fit_preference": "regular",
            "background_action": "preserve"
        }
        resp = requests.post(f"{API_URL}/tryon", files=files, data=data)
    
    res_data = resp.json()
    print(res_data)
    
    if "job_id" in res_data:
        job_id = res_data["job_id"]
        print(f"Polling job status for {job_id}...")
        for _ in range(5):
            status_resp = requests.get(f"{API_URL}/status/{job_id}")
            print(status_resp.json())
            if status_resp.json().get("status") in ["completed", "failed"]:
                break
            time.sleep(2)

if __name__ == "__main__":
    try:
        test_health()
        test_semantic_analysis()
        test_generation()
        test_tryon()
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
