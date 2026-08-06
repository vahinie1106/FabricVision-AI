import asyncio
from pathlib import Path
from backend_api.services.job_manager import job_manager
from backend_api.services.generation_service import model_manager

async def process_tryon(job_id: str, garment_image_path: str, person_image_path: str, fit_preference: str, background_action: str):
    """
    Background worker for CatVTON try-on.
    """
    try:
        job_manager.update_job(job_id, status="processing", progress=10, current_step="Loading CatVTON pipeline...")
        
        from src.features.virtual_tryon.tryon_pipeline import VirtualTryOnPipeline
        from src.features.virtual_tryon.models import GarmentConditioningInput, PersonConditioningInput
        from PIL import Image
        
        # We need a model_loader adapter or pass model_manager depending on implementation.
        # The existing TryOn pipeline expects a CatVTONModelLoader but we'll instantiate VirtualTryOnPipeline
        pipeline = VirtualTryOnPipeline()
        
        job_manager.update_job(job_id, progress=30, current_step="Preparing conditions...")
        
        loop = asyncio.get_running_loop()
        
        def run_sync_tryon():
            # Load images
            g_img = Image.open(garment_image_path).convert("RGB")
            p_img = Image.open(person_image_path).convert("RGB")
            
            garment_in = GarmentConditioningInput(garment_image=g_img)
            person_in = PersonConditioningInput(person_image=p_img)
            
            return pipeline.run(person_in, garment_in)
            
        job_manager.update_job(job_id, progress=50, current_step="Mapping garment to person (CatVTON)...")
        result = await loop.run_in_executor(None, run_sync_tryon)
        
        img_path = getattr(result, "image_path", None) or getattr(result, "output_path", None) or (result.get("image_path") if isinstance(result, dict) else None)
        result_url_val = None
        if img_path:
            p = Path(img_path)
            try:
                from backend_api.config.settings import settings
                import shutil
                rel_path = p.relative_to(settings.OUTPUT_DIR)
                result_url_val = f"/outputs/{rel_path.as_posix()}"
            except ValueError:
                from backend_api.config.settings import settings
                import shutil
                filename = p.name
                dest = settings.OUTPUT_DIR / filename
                if p.exists() and not dest.exists():
                    shutil.copy2(p, dest)
                result_url_val = f"/outputs/{filename}"

        job_manager.update_job(
            job_id, 
            status="completed", 
            progress=100, 
            current_step="Try-on successful",
            result_url=result_url_val
        )
        
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e), current_step="Failed")

