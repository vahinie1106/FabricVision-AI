from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Request

from backend_api.schemas.response_models import JobCreationResponse, JobStatusResponse
from backend_api.utils.file_storage import save_upload_file
from backend_api.services.job_manager import job_manager
from backend_api.services.generation_service import (
    process_generation,
    resolve_incoming_generation_mode,
)

router = APIRouter()

@router.post("/generate", response_model=JobCreationResponse)
async def generate_garment(
    request: Request,
    background_tasks: BackgroundTasks,
    fabric_image: UploadFile = File(...),
    garment_type: str = Form(...),
    fit: str = Form(...),
    style: str = Form(...),
    gender: str = Form("women"),
    season: str = Form("summer"),
    occasion: str = Form("casual"),
    fabric: str = Form("cotton"),
    material: str = Form("cotton"),
    texture: str = Form("smooth"),
    color: str = Form("white"),
    sleeve: str = Form("short"),
    neckline: str = Form("round"),
    generation_mode: Optional[str] = Form(None),
):
    """
    Submit a custom garment generation job (FLUX.1-Kontext image-conditioned).

    generation_mode: preview | standard | production
    (also accepts UI labels Preview / Standard / Production).
    Required — never silently defaults to Standard / 3 steps.
    Also accepted via query ``?generation_mode=`` or header
    ``X-Fabricvision-Generation-Mode`` (Kaggle multipart backup).
    """
    header_mode = request.headers.get("x-fabricvision-generation-mode")
    query_mode = request.query_params.get("generation_mode")
    print("[API QUALITY DEBUG]", flush=True)
    print(f"[API QUALITY DEBUG] generation_mode_form={generation_mode!r}", flush=True)
    print(f"[API QUALITY DEBUG] generation_mode_header={header_mode!r}", flush=True)
    print(f"[API QUALITY DEBUG] generation_mode_query={query_mode!r}", flush=True)
    try:
        generation_mode = resolve_incoming_generation_mode(
            generation_mode, header_mode, query_mode
        )
    except ValueError as exc:
        print("[API QUALITY DEBUG] generation_mode=<missing>", flush=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    print(f"[API QUALITY DEBUG] generation_mode={generation_mode}", flush=True)
    print(
        f"[QUALITY DEBUG] api_received_generation_mode={generation_mode}",
        flush=True,
    )
    try:
        fabric_path = save_upload_file(fabric_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    job_id = job_manager.create_job()
    
    background_tasks.add_task(
        process_generation,
        job_id,
        fabric_path,
        garment_type,
        fit,
        style,
        gender,
        season,
        occasion,
        fabric,
        material,
        texture,
        color,
        sleeve,
        neckline,
        generation_mode,
    )
    
    return JobCreationResponse(job_id=job_id)

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_generation_status(job_id: str):
    """
    Poll the status of a generation job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=(
                "Job not found "
                f"(job_id={job_id}, pid={__import__('os').getpid()}). "
                "If generation was in progress, the backend process likely restarted "
                "— retry Generate on a single uvicorn worker without --reload."
            ),
        )
        
    return job
