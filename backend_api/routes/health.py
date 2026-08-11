from fastapi import APIRouter
from backend_api.schemas.response_models import HealthResponse
from backend_api.config.settings import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION
    )


@router.get("/flux-status")
async def flux_status():
    """
    API-process FLUX residency (not the run_kaggle parent prefetch).

    Generate should only proceed when ready/in_memory is true in THIS process.
    """
    from backend_api.services.flux_warmup import get_warmup_status
    import os

    status = get_warmup_status()
    status["api_pid"] = os.getpid()
    return status

