from pydantic import BaseModel
from typing import Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    version: str

class JobCreationResponse(BaseModel):
    job_id: str
    status: str = "queued"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: int = 0
    current_step: Optional[str] = None
    # Authoritative lifecycle stage (e.g. generating, saving, completed).
    # Frontend must prefer this over inferred UI percentages.
    stage: Optional[str] = None
    result_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Machine-stable failure classification for UI mapping (failed jobs only).
    error_type: Optional[str] = None
    failed_stage: Optional[str] = None

class SemanticAnalysisResponse(BaseModel):
    status: str
    metadata: Dict[str, Any]
    confidence: Optional[float] = None
    error: Optional[str] = None
