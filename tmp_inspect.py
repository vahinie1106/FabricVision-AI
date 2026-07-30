from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from src.dataset_management.dataset_manager import DatasetManagementConfig, DatasetManager

with TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    dataset_root = tmp_path / 'datasets'
    output_root = tmp_path / 'prepared'
    reports_dir = tmp_path / 'reports'
    dataset_root.mkdir(parents=True, exist_ok=True)
    men_product = dataset_root / 'men' / 'shirts' / 'prod_001'
    men_product.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (128, 160), color='red').save(men_product / 'front.jpg')
    women_product = dataset_root / 'women' / 'dresses' / 'prod_002'
    women_product.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (160, 200), color='blue').save(women_product / 'front.png')
    config = DatasetManagementConfig(dataset_root=str(dataset_root), output_root=str(output_root), reports_dir=str(reports_dir), supported_formats=['.jpg', '.jpeg', '.png', '.webp'])
    manager = DatasetManager(config=config)
    result = manager.run()
    print('products', result['products_indexed'])
    print('valid', result['valid_products'])
    print('target1', (output_root / 'garments' / 'men' / 'upperwear' / 'prod_001').exists())
    print('target2', (output_root / 'garments' / 'women' / 'dresses' / 'prod_002').exists())
    print('tree', [str(p.relative_to(tmp_path)) for p in sorted(output_root.rglob('*'))])
