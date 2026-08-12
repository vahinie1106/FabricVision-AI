from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from backend_api.schemas.response_models import JobCreationResponse
from backend_api.utils.file_storage import save_upload_file
from backend_api.services.job_manager import job_manager
from backend_api.services.tryon_service import process_tryon

router = APIRouter()

@router.post("/tryon", response_model=JobCreationResponse)
async def virtual_tryon(
    background_tasks: BackgroundTasks,
    garment_image: UploadFile = File(...),
    person_image: UploadFile = File(...),
    fit_preference: str = Form(...),
    background_action: str = Form(...),
    garment_type: str = Form("garment"),
):
    """
    Submit a virtual try-on job.
    """
    try:
        g_path = save_upload_file(garment_image)
        p_path = save_upload_file(person_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    job_id = job_manager.create_job()
    
    background_tasks.add_task(
        process_tryon,
        job_id,
        g_path,
        p_path,
        fit_preference,
        background_action,
        garment_type,
    )
    
    return JobCreationResponse(job_id=job_id)
