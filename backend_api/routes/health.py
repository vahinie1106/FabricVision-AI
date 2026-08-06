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
