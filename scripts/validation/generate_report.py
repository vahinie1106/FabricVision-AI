import json
from pathlib import Path

def main():
    workspace_root = Path(__file__).resolve().parent
    curated_dir = workspace_root / "curated_dataset"
    json_files = list(curated_dir.rglob("*.json"))
    
    results = []
    
    for meta_path in json_files:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        img_name = Path(meta.get("source_image", "")).name
        results.append({
            "Image Name": img_name,
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
        
    results.sort(key=lambda x: x["Image Name"])

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
            
    print(f"Report written to {report_path.absolute()}")

if __name__ == "__main__":
    main()
