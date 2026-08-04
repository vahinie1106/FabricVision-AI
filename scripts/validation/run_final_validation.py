import json
from pathlib import Path

from src.preprocessing.run_preprocessing import main as run_preprocessing
from src.semantic_analysis.pipeline import SemanticAnalysisConfig, SemanticAnalysisPipeline

def main():
    print("Running Preprocessing Pipeline...")
    run_preprocessing()
    
    print("\nLoading Semantic Analysis Pipeline...")
    workspace_root = Path(__file__).resolve().parent
    config = SemanticAnalysisConfig()
    config.config_dir = str(workspace_root / "configs")
    config.output_root = str(workspace_root / "curated_dataset")
    config.config_path = str(workspace_root / "configs" / "semantic_analysis_config.yaml")
    
    pipeline = SemanticAnalysisPipeline(config=config)
    
    processed_dir = Path("data/processed_validation")
    images = list(processed_dir.rglob("*.jpg"))
    
    results = []
    
    for idx, img in enumerate(images, 1):
        print(f"Processing {idx}/{len(images)}: {img.name}")
        res = pipeline.run(str(img))
        
        status = res.get("status")
        if status == "completed":
            meta_path = Path(res.get("metadata_path"))
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                
            results.append({
                "Image Name": img.name,
                "Gender": meta.get("garment_identity", {}).get("gender"),
                "Category": meta.get("classification", {}).get("category"),
                "Subcategory": meta.get("classification", {}).get("subcategory"),
                "Material": meta.get("physical_attributes", {}).get("material"),
                "Pattern": meta.get("visual_attributes", {}).get("patterns"),
                "Primary Color": meta.get("visual_attributes", {}).get("colors"),
                "Confidence": meta.get("ai_analysis", {}).get("confidence"),
                "Status": "SUCCESS",
                "Path": str(meta_path.relative_to(workspace_root))
            })
        else:
            print(f"FAILED: {res.get('issues')}")
            results.append({
                "Image Name": img.name,
                "Gender": "-",
                "Category": "-",
                "Subcategory": "-",
                "Material": "-",
                "Pattern": "-",
                "Primary Color": "-",
                "Confidence": "-",
                "Status": f"FAILED: {res.get('issues')}",
                "Path": "-"
            })

    # Generate Markdown Report
    report_path = Path("Final_Validation_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Final Validation Report\n\n")
        f.write("| Image Name | Gender | Category | Subcategory | Material | Pattern | Primary Color | Confidence | Status | Organization Path |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            pat = ", ".join(r["Pattern"]) if isinstance(r["Pattern"], list) else r["Pattern"]
            col = ", ".join(r["Primary Color"]) if isinstance(r["Primary Color"], list) else r["Primary Color"]
            f.write(f"| {r['Image Name']} | {r['Gender']} | {r['Category']} | {r['Subcategory']} | {r['Material']} | {pat} | {col} | {r['Confidence']} | {r['Status']} | {r['Path']} |\n")
            
    print(f"\nReport written to {report_path.absolute()}")

if __name__ == "__main__":
    main()
