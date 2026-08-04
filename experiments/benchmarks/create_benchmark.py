import random
import shutil
import json
from pathlib import Path

def get_unique_name(src_path):
    # e.g., datasets/fashion_garments/img/WOMEN/Dresses/id_00003954/01_4_full.jpg
    # to id_00003954_01_4_full.jpg
    parent = src_path.parent.name
    return f"{parent}_{src_path.name}"

def main():
    root = Path(__file__).resolve().parent
    men_source = root / "datasets" / "fashion_garments" / "img" / "MEN"
    women_source = root / "datasets" / "fashion_garments" / "img" / "WOMEN"
    
    out_men = root / "datasets" / "scale_testing" / "sample_500" / "men"
    out_women = root / "datasets" / "scale_testing" / "sample_500" / "women"
    
    # 1. Clean up old incorrect folders
    for d in [out_men, out_women]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Gather all images
    men_imgs = list(men_source.rglob("*.jpg")) + list(men_source.rglob("*.jpeg")) + list(men_source.rglob("*.png"))
    women_imgs = list(women_source.rglob("*.jpg")) + list(women_source.rglob("*.jpeg")) + list(women_source.rglob("*.png"))
    
    # 3. Shuffle
    random.seed(42)
    random.shuffle(men_imgs)
    random.shuffle(women_imgs)
    
    records = []
    
    # 4. Copy exactly 250 unique men
    men_count = 0
    men_names = set()
    for src in men_imgs:
        if men_count >= 250:
            break
        unique_name = get_unique_name(src)
        if unique_name in men_names:
            continue
        dest = out_men / unique_name
        shutil.copy(src, dest)
        records.append({
            "original_source_path": str(src),
            "new_copied_path": str(dest),
            "gender": "men",
            "filename": unique_name,
            "selection_timestamp": __import__("time").time()
        })
        men_names.add(unique_name)
        men_count += 1
        
    # 5. Copy exactly 250 unique women
    women_count = 0
    women_names = set()
    for src in women_imgs:
        if women_count >= 250:
            break
        unique_name = get_unique_name(src)
        if unique_name in women_names:
            continue
        dest = out_women / unique_name
        shutil.copy(src, dest)
        records.append({
            "original_source_path": str(src),
            "new_copied_path": str(dest),
            "gender": "women",
            "filename": unique_name,
            "selection_timestamp": __import__("time").time()
        })
        women_names.add(unique_name)
        women_count += 1
        
    # 6. Save dataset_record.json
    record_path = root / "datasets" / "scale_testing" / "sample_500" / "dataset_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(records),
            "gender_split": {"men": men_count, "women": women_count},
            "images": records
        }, f, indent=2)
        
    print(f"Men images: {men_count}")
    print(f"Women images: {women_count}")
    print(f"Total images: {len(records)}")
    print(f"Folder location: {root / 'datasets' / 'scale_testing' / 'sample_500'}")

if __name__ == "__main__":
    main()
