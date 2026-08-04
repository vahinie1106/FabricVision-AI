import random
import shutil
from pathlib import Path
import json

def main():
    root = Path(__file__).resolve().parent
    men_source = root / "datasets" / "fashion_garments" / "img" / "MEN"
    women_source = root / "datasets" / "fashion_garments" / "img" / "WOMEN"
    
    out_men = root / "datasets" / "scale_testing" / "sample_500" / "men"
    out_women = root / "datasets" / "scale_testing" / "sample_500" / "women"
    
    for d in [out_men, out_women]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    men_imgs = list(men_source.rglob("*.jpg")) + list(men_source.rglob("*.jpeg")) + list(men_source.rglob("*.png"))
    women_imgs = list(women_source.rglob("*.jpg")) + list(women_source.rglob("*.jpeg")) + list(women_source.rglob("*.png"))
    
    sel_men = random.sample(men_imgs, min(250, len(men_imgs)))
    sel_women = random.sample(women_imgs, min(250, len(women_imgs)))
    
    records = []
    
    for src in sel_men:
        dest = out_men / src.name
        shutil.copy(src, dest)
        records.append({"original": str(src), "copied": str(dest), "gender": "men"})
        
    for src in sel_women:
        dest = out_women / src.name
        shutil.copy(src, dest)
        records.append({"original": str(src), "copied": str(dest), "gender": "women"})
        
    record_path = root / "datasets" / "scale_testing" / "sample_500" / "dataset_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(records),
            "gender_split": {"men": len(sel_men), "women": len(sel_women)},
            "images": records
        }, f, indent=2)
        
    print(f"Copied {len(sel_men)} men images to {out_men}")
    print(f"Copied {len(sel_women)} women images to {out_women}")
    print(f"Total: {len(records)} images.")
    print(f"Record saved to {record_path}")

if __name__ == "__main__":
    main()
