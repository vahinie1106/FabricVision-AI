import json
from pathlib import Path
from src.semantic_analysis.pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline

def main():
    workspace_root = Path(__file__).resolve().parent
    config = SemanticAnalysisConfig()
    config.config_dir = str(workspace_root / "configs")
    config.output_root = str(workspace_root / "curated_dataset")
    config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    
    pipeline = SemanticAnalysisPipeline(config=config)
    img = Path("data/processed_validation/women/03_1_front.jpg")
    
    print(f"Rerunning {img.name}...")
    res = pipeline.run(str(img))
    
    status = res.get("status")
    if status == "completed":
        print("SUCCESS!")
    else:
        print(f"FAILED: {res.get('issues')}")

if __name__ == "__main__":
    main()
