import asyncio
from pathlib import Path
from PIL import Image
import shutil
from backend_api.config.settings import settings
from backend_api.services.job_manager import job_manager
from src.common.models.model_manager import ModelManager

# Initialize the ModelManager globally so VRAM is shared
model_manager = ModelManager()

async def process_generation(
    job_id: str,
    fabric_image_path: str,
    garment_type: str,
    fit: str,
    style: str,
    gender: str = "women",
    season: str = "summer",
    occasion: str = "casual",
    fabric: str = "cotton",
    material: str = "cotton",
    texture: str = "smooth",
    color: str = "white",
    sleeve: str = "short",
    neckline: str = "round",
):
    """
    Background worker for FLUX generation.
    """
    try:
        job_manager.update_job(job_id, status="processing", progress=10, current_step="Loading FLUX pipeline...")
        
        # Correct import
        from src.features.custom_generator.pipeline.garment_generation_pipeline import GarmentGenerationPipeline
        
        pipeline = GarmentGenerationPipeline()
        
        job_manager.update_job(job_id, progress=30, current_step="Analyzing fabric texture...")
        
        loop = asyncio.get_running_loop()
        
        def run_sync_generation():
            ref_img = Image.open(fabric_image_path).convert("RGB")
            fabric_metadata = {
                "material": material.lower(),
                "fabric": fabric.lower(),
                "texture": texture.lower(),
                "dominant_colors": [color.lower()],
                "color": color.lower(),
                "style": style.lower(),
                "occasion": occasion.lower(),
                "season": season.lower(),
                "fit": fit.lower(),
            }
            user_customization = {
                "gender": gender.lower(),
                "garment_type": garment_type.lower().replace(" ", "_"),
                "material": material.lower(),
                "fabric": fabric.lower(),
                "texture": texture.lower(),
                "color": color.lower(),
                "neckline": neckline.lower().replace(" ", "_"),
                "sleeve": sleeve.lower().replace(" ", "_"),
                "occasion": occasion.lower(),
                "season": season.lower(),
                "fit": fit.lower(),
                "style": style.lower(),
            }
            return pipeline.run(
                fabric_metadata=fabric_metadata,
                user_customization=user_customization,
                reference_image=ref_img
            )
            
        job_manager.update_job(job_id, progress=50, current_step="Synthesizing garment (FLUX)...")
        
        result = await loop.run_in_executor(None, run_sync_generation)
        
        img_path = result.get("image_path") or result.get("output_path")
        result_url_val = None
        if img_path:
            p = Path(img_path)
            try:
                rel_path = p.relative_to(settings.OUTPUT_DIR)
                result_url_val = f"/outputs/{rel_path.as_posix()}"
            except ValueError:
                filename = p.name
                dest = settings.OUTPUT_DIR / filename
                if p.exists() and not dest.exists():
                    shutil.copy2(p, dest)
                result_url_val = f"/outputs/{filename}"

        job_manager.update_job(
            job_id, 
            status="completed", 
            progress=100, 
            current_step="Generation successful",
            result_url=result_url_val,
            metadata={
                "category": garment_type,
                "fabric": fabric.capitalize(),
                "styleAffinity": style,
                "confidenceScore": 0.95
            }
        )
        
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e), current_step="Failed")

