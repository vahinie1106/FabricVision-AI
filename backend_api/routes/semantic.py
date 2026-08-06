import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend_api.schemas.response_models import SemanticAnalysisResponse
from backend_api.utils.file_storage import save_upload_file, cleanup_file
from backend_api.services.semantic_service import process_semantic_analysis

router = APIRouter()

@router.post("/analyze", response_model=SemanticAnalysisResponse)
async def analyze_garment(garment_image: UploadFile = File(...)):
    """
    Analyze a garment image synchronously using Qwen2.5-VL.
    """
    try:
        image_path = save_upload_file(garment_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        result = await process_semantic_analysis(image_path)
        
        if result["status"] == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))
            
        return SemanticAnalysisResponse(
            status="completed",
            metadata=result["metadata"],
            confidence=result.get("confidence")
        )
    finally:
        cleanup_file(image_path)
