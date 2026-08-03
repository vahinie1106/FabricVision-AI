from pathlib import Path
import shutil

root = Path('datasets/fashion_garments/img')
men_root = root / 'MEN'
women_root = root / 'WOMEN'
out_root = Path('data/processed/semantic_analysis_sample')
for p in [out_root / 'men', out_root / 'women']:
    p.mkdir(parents=True, exist_ok=True)

men_files = []
for subdir in sorted(men_root.iterdir()):
    if subdir.is_dir():
        men_files.extend(sorted(subdir.glob('*')))
men_files = [p for p in men_files if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}][:10]

women_files = []
for subdir in sorted(women_root.iterdir()):
    if subdir.is_dir():
        women_files.extend(sorted(subdir.glob('*')))
women_files = [p for p in women_files if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}][:10]

for src in men_files + women_files:
    target_dir = out_root / ('men' if 'MEN' in str(src) else 'women')
    shutil.copy2(src, target_dir / src.name)

print('men', len(men_files))
print('women', len(women_files))
print('sample_dir', out_root)
for p in sorted(out_root.glob('*/*')):
    print(p)
