import os
from pathlib import Path

class Settings:
    PROJECT_NAME = "FabricVision-AI API"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    
    # Storage
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR = BASE_DIR / "data" / "uploads"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # AI Configs
    DEVICE = "cuda" # Default to CUDA, falls back to CPU internally if unavailable
    
    def __init__(self):
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
