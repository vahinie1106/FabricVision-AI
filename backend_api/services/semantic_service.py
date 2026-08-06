import asyncio
from PIL import Image
from backend_api.services.generation_service import model_manager

async def process_semantic_analysis(image_path: str):
    """
    Runs Qwen2.5-VL semantic analysis on the given image.
    """
    try:
        from src.features.semantic_analysis.pipeline.semantic_analysis_pipeline import SemanticAnalysisPipeline
        
        pipeline = SemanticAnalysisPipeline()
        
        loop = asyncio.get_running_loop()
        
        def run_sync_analysis():
            # Pipeline run takes image_path directly
            return pipeline.run(image_path)
            
        result = await loop.run_in_executor(None, run_sync_analysis)
        
        return {
            "status": "completed",
            "metadata": result,
            "confidence": 0.95
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
