import sys
import time
from pathlib import Path

from src.semantic_analysis.pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline

def load_config() -> SemanticAnalysisConfig:
    workspace_root = Path(__file__).resolve().parent
    config = SemanticAnalysisConfig()
    config.config_dir = str(workspace_root / "configs")
    config.output_root = str(workspace_root / "curated_dataset")
    config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    return config

def main():
    print("Loading pipeline...")
    config = load_config()
    pipeline = SemanticAnalysisPipeline(config=config)
    print("Pipeline loaded.")
    
    input_dir = Path("data/processed_validation")
    images = list(input_dir.rglob("*.jpg"))
    print(f"Found {len(images)} images to process.")
    
    results = []
    
    for img in images:
        print(f"Processing {img}...")
        start = time.perf_counter()
        result = pipeline.run(str(img))
        elapsed = time.perf_counter() - start
        
        status = result.get("status")
        if status == "completed":
            print(f"SUCCESS {img} - {elapsed:.2f}s")
            results.append({
                "image": img.name,
                "status": "Pass",
                "metadata": result.get("metadata_path")
            })
        else:
            print(f"FAILED {img} - {result.get('issues')}")
            results.append({
                "image": img.name,
                "status": "Fail",
                "issues": result.get("issues")
            })

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
