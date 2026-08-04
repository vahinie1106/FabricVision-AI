import random
import shutil
import json
import time
from pathlib import Path

def get_used_images(root: Path):
    used = set()
    # Check sample_10
    val_10_dir = root / "datasets" / "validation" / "sample_10"
    if val_10_dir.exists():
        for img in val_10_dir.rglob("*.jpg"):
            used.add(img.name)
            
    # Check sample_500 record
    record_500 = root / "datasets" / "scale_testing" / "sample_500" / "dataset_record.json"
    if record_500.exists():
        with open(record_500, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("images", []):
                used.add(item["filename"])
                # Also original name just in case
                used.add(Path(item["original_source_path"]).name)
                
    return used

def get_unique_name(src_path):
    parent = src_path.parent.name
    return f"{parent}_{src_path.name}"

def main():
    root = Path(__file__).resolve().parent
    men_source = root / "datasets" / "fashion_garments" / "img" / "MEN"
    women_source = root / "datasets" / "fashion_garments" / "img" / "WOMEN"
    
    out_dir = root / "datasets" / "validation" / "sample_50"
    out_men = out_dir / "men"
    out_women = out_dir / "women"
    
    # Setup directories
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_men.mkdir(parents=True, exist_ok=True)
    out_women.mkdir(parents=True, exist_ok=True)
    
    used_names = get_used_images(root)
    
    men_imgs = list(men_source.rglob("*.jpg"))
    women_imgs = list(women_source.rglob("*.jpg"))
    
    random.seed(123)
    random.shuffle(men_imgs)
    random.shuffle(women_imgs)
    
    records = []
    
    men_count = 0
    men_names = set()
    for src in men_imgs:
        if men_count >= 25:
            break
        unique_name = get_unique_name(src)
        if unique_name in used_names or unique_name in men_names or src.name in used_names:
            continue
        dest = out_men / unique_name
        shutil.copy(src, dest)
        records.append({
            "original_path": str(src),
            "copied_path": str(dest),
            "gender": "men",
            "garment_id": src.parent.name,
            "filename": unique_name,
            "timestamp": time.time()
        })
        men_names.add(unique_name)
        men_count += 1
        
    women_count = 0
    women_names = set()
    for src in women_imgs:
        if women_count >= 25:
            break
        unique_name = get_unique_name(src)
        if unique_name in used_names or unique_name in women_names or src.name in used_names:
            continue
        dest = out_women / unique_name
        shutil.copy(src, dest)
        records.append({
            "original_path": str(src),
            "copied_path": str(dest),
            "gender": "women",
            "garment_id": src.parent.name,
            "filename": unique_name,
            "timestamp": time.time()
        })
        women_names.add(unique_name)
        women_count += 1
        
    record_path = out_dir / "dataset_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(records),
            "men": men_count,
            "women": women_count,
            "images": records
        }, f, indent=2)
        
    print(f"Dataset created: {len(records)} images in {out_dir}")

if __name__ == "__main__":
    main()
