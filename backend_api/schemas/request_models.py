from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# These models are primarily used for JSON bodies.
# Note: For file uploads (multipart/form-data), FastAPI uses Form() parameters instead of Pydantic models.

class JobStatusRequest(BaseModel):
    job_id: str

class ErrorDetail(BaseModel):
    error: str
    message: str
