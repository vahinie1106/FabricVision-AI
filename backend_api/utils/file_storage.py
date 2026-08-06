import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException

from backend_api.config.settings import settings

def save_upload_file(upload_file: UploadFile) -> str:
    """Save an uploaded file to the temporary upload directory and return the absolute path."""
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in upload.")
        
    ext = os.path.splitext(upload_file.filename)[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")
        
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = settings.UPLOAD_DIR / unique_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        upload_file.file.close()
        
    return str(file_path)

def cleanup_file(file_path: str) -> None:
    """Remove a temporary file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to cleanup file {file_path}: {e}")
