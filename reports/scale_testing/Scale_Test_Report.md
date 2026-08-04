# Scale Test Report

## 1. Test Information
- **Date**: 2026-08-04
- **Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Model**: Qwen2.5-VL-3B-Instruct
- **Quantization Type**: 4-bit NF4

## 2. Dataset Information
- **Total Images**: 119
- **Male Images**: 250
- **Female Images**: 250
- **Source Datasets**: `datasets/fashion_garments`

## 3. Performance Metrics
- **Total Execution Time**: 3538.49 seconds
- **Average Seconds/Image**: 29.74 seconds
- **GPU Memory Usage**: 2600.70 MiB

## 4. Metadata Quality
| Image Name | Detected Gender | Category | Subcategory | Confidence Score | Validation Status | Output Path |
|---|---|---|---|---|---|---|
| 01_1_front.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.patterns', 'message': 'Value not allowed: dots'}] | - |
| 01_2_side.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: dark blue'}] | - |
| 01_3_back.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.patterns', 'message': 'Value not allowed: spotted'}] | - |
| 01_4_full.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: checkered'}] | - |
| 01_6_flat.jpg | men | upper_wear | jacket | 0.95 | SUCCESS | curated_dataset\men\upper_wear\jacket\01_6_flat.json |
| 01_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky blue'}] | - |
| 02_1_front.jpg | men | upper_wear | hoodie | 0.95 | SUCCESS | curated_dataset\men\upper_wear\hoodie\02_1_front.json |
| 02_2_side.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: camouflage'}] | - |
| 02_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 02_4_full.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\02_4_full.json |
| 02_6_flat.jpg | unisex | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\unisex\upper_wear\shirt\02_6_flat.json |
| 02_7_additional.jpg | men | upper_wear | hoodie | 0.95 | SUCCESS | curated_dataset\men\upper_wear\hoodie\02_7_additional.json |
| 03_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['navy blue']"}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | - |
| 03_2_side.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}] | - |
| 03_3_back.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: navy blue'}] | - |
| 03_4_full.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\03_4_full.json |
| 03_7_additional.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\03_7_additional.json |
| 04_1_front.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: navy blue'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: white polka dots'}] | - |
| 04_2_side.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\04_2_side.json |
| 04_3_back.jpg | men | upper_wear | hoodie | 0.95 | SUCCESS | curated_dataset\men\upper_wear\hoodie\04_3_back.json |
| 04_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 04_6_flat.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': "Invalid values: ['unknown']"}] | - |
| 04_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: white'}] | - |
| 05_1_front.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\05_1_front.json |
| 05_2_side.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 05_3_back.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | - |
| 05_4_full.jpg | men | upper_wear | hoodie | 0.95 | SUCCESS | curated_dataset\men\upper_wear\hoodie\05_4_full.json |
| 05_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: maroon'}] | - |
| 06_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 06_2_side.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\06_2_side.json |
| 06_3_back.jpg | men | upper_wear | sweater | 0.95 | SUCCESS | curated_dataset\men\upper_wear\sweater\06_3_back.json |
| 06_4_full.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': "Invalid values: ['teal']"}] | - |
| 06_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 07_2_side.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\07_2_side.json |
| 07_3_back.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\07_3_back.json |
| 07_4_full.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 07_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 08_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: racerback'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 08_2_side.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\08_2_side.json |
| 08_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: t-shirt'}] | - |
| 08_4_full.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\08_4_full.json |
| 08_7_additional.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\08_7_additional.json |
| 09_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 09_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 09_3_back.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 09_4_full.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive'}] | - |
| 09_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | - |
| 10_1_front.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\10_1_front.json |
| 10_6_flat.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky blue'}] | - |
| 10_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 11_1_front.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\11_1_front.json |
| 11_3_back.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\11_3_back.json |
| 11_4_full.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\11_4_full.json |
| 12_1_front.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 12_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: turtleneck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: burgundy'}] | - |
| 13_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 13_2_side.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\13_2_side.json |
| 13_3_back.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive_green'}] | - |
| 14_6_flat.jpg | unisex | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\unisex\upper_wear\shirt\14_6_flat.json |
| 15_2_side.jpg | men | upper_wear | hoodie | 0.95 | SUCCESS | curated_dataset\men\upper_wear\hoodie\15_2_side.json |
| 17_1_front.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}] | - |
| 20_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 27_1_front.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}] | - |
| 28_6_flat.jpg | unisex | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\unisex\upper_wear\shirt\28_6_flat.json |
| 30_1_front.jpg | men | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\men\upper_wear\shirt\30_1_front.json |
| 01_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.patterns', 'message': "Invalid values: ['triangle']"}] | - |
| 01_2_side.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | - |
| 01_3_back.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: plaid'}] | - |
| 01_4_full.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\01_4_full.json |
| 01_7_additional.jpg | women | upper_wear | jacket | 0.95 | SUCCESS | curated_dataset\women\upper_wear\jacket\01_7_additional.json |
| 02_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}] | - |
| 02_2_side.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | - |
| 02_3_back.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\02_3_back.json |
| 02_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank_top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 02_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: velvet'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive green'}] | - |
| 03_1_front.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}] | - |
| 03_2_side.jpg | women | upper_wear | sweater | 0.95 | SUCCESS | curated_dataset\women\upper_wear\sweater\03_2_side.json |
| 03_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short_dresses'}] | - |
| 03_4_full.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\03_4_full.json |
| 03_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 04_1_front.jpg | - | - | - | - | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: animal print'}] | - |
| 04_2_side.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\04_2_side.json |
| 04_3_back.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\04_3_back.json |
| 04_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: maxi dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strappy back'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: vertical stripes'}] | - |
| 04_6_flat.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: maxi dress'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strappy'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['navy blue']"}] | - |
| 04_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: skater dress'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 05_1_front.jpg | women | upper_wear | sweater | 0.95 | SUCCESS | curated_dataset\women\upper_wear\sweater\05_1_front.json |
| 05_2_side.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\05_2_side.json |
| 05_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: sweater dress'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: three-quarter'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 05_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: open back'}] | - |
| 06_2_side.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\06_2_side.json |
| 06_4_full.jpg | women | upper_wear | jacket | 0.95 | SUCCESS | curated_dataset\women\upper_wear\jacket\06_4_full.json |
| 07_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: t-shirt dress'}] | - |
| 07_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 07_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: strapless dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strapless'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 07_7_additional.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\07_7_additional.json |
| 08_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 08_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 08_3_back.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: lace'}] | - |
| 08_4_full.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\08_4_full.json |
| 09_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 09_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 09_7_additional.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\09_7_additional.json |
| 10_2_side.jpg | - | - | - | - | FAILED: [{'field': 'style.season', 'message': 'Value not allowed: fall'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['olive green']"}] | - |
| 10_3_back.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'style.season', 'message': 'Value not allowed: fall'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive green'}] | - |
| 11_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: crop top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | - |
| 12_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 14_7_additional.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\14_7_additional.json |
| 15_1_front.jpg | - | - | - | - | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky_blue'}] | - |
| 16_1_front.jpg | women | upper_wear | blouse | 0.95 | SUCCESS | curated_dataset\women\upper_wear\blouse\16_1_front.json |
| 20_4_full.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 21_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | - |
| 22_7_additional.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | - |
| 28_3_back.jpg | women | upper_wear | sweater | 0.95 | SUCCESS | curated_dataset\women\upper_wear\sweater\28_3_back.json |
| 30_3_back.jpg | - | - | - | - | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | - |
| 31_2_side.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 49_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | - |
| 52_1_front.jpg | - | - | - | - | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | - |
| 64_1_front.jpg | women | upper_wear | shirt | 0.95 | SUCCESS | curated_dataset\women\upper_wear\shirt\64_1_front.json |

## 5. Failure Analysis
| Image Name | Error | Module Responsible | Fix Applied |
|---|---|---|---|
| 01_1_front.jpg | FAILED: [{'field': 'visual_attributes.patterns', 'message': 'Value not allowed: dots'}] | MetadataValidator | None |
| 01_2_side.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: dark blue'}] | MetadataValidator | None |
| 01_3_back.jpg | FAILED: [{'field': 'visual_attributes.patterns', 'message': 'Value not allowed: spotted'}] | MetadataValidator | None |
| 01_4_full.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: checkered'}] | MetadataValidator | None |
| 01_7_additional.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky blue'}] | MetadataValidator | None |
| 02_2_side.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: camouflage'}] | MetadataValidator | None |
| 02_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 03_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['navy blue']"}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 03_2_side.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 03_3_back.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: navy blue'}] | MetadataValidator | None |
| 04_1_front.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: navy blue'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: white polka dots'}] | MetadataValidator | None |
| 04_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 04_6_flat.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': "Invalid values: ['unknown']"}] | MetadataValidator | None |
| 04_7_additional.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: white'}] | MetadataValidator | None |
| 05_2_side.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 05_3_back.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 05_7_additional.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: maroon'}] | MetadataValidator | None |
| 06_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 06_4_full.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': "Invalid values: ['teal']"}] | MetadataValidator | None |
| 06_7_additional.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 07_4_full.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 07_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 08_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: racerback'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 08_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: t-shirt'}] | MetadataValidator | None |
| 09_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 09_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 09_3_back.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 09_4_full.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive'}] | MetadataValidator | None |
| 09_7_additional.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.drape', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.flexibility', 'message': 'Value not allowed: unknown'}, {'field': 'fabric_behaviour.thickness', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 10_6_flat.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky blue'}] | MetadataValidator | None |
| 10_7_additional.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 12_1_front.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 12_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: turtleneck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: burgundy'}] | MetadataValidator | None |
| 13_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 13_3_back.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive_green'}] | MetadataValidator | None |
| 17_1_front.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 20_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 27_1_front.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 01_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.patterns', 'message': "Invalid values: ['triangle']"}] | MetadataValidator | None |
| 01_2_side.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 01_3_back.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: plaid'}] | MetadataValidator | None |
| 02_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}] | MetadataValidator | None |
| 02_2_side.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}] | MetadataValidator | None |
| 02_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank_top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 02_7_additional.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: velvet'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive green'}] | MetadataValidator | None |
| 03_1_front.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}] | MetadataValidator | None |
| 03_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short_dresses'}] | MetadataValidator | None |
| 03_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 04_1_front.jpg | FAILED: [{'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: black, white'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: animal print'}] | MetadataValidator | None |
| 04_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: maxi dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strappy back'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: vertical stripes'}] | MetadataValidator | None |
| 04_6_flat.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: maxi dress'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strappy'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['navy blue']"}] | MetadataValidator | None |
| 04_7_additional.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: skater dress'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 05_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: sweater dress'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: three-quarter'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 05_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: open back'}] | MetadataValidator | None |
| 07_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: t-shirt dress'}] | MetadataValidator | None |
| 07_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 07_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: strapless dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: strapless'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 08_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 08_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 08_3_back.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}, {'field': 'visual_attributes.patterns', 'message': 'Value not allowed: lace'}] | MetadataValidator | None |
| 09_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: short dress'}, {'field': 'physical_attributes.material', 'message': 'Value not allowed: unknown'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 09_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
| 10_2_side.jpg | FAILED: [{'field': 'style.season', 'message': 'Value not allowed: fall'}, {'field': 'visual_attributes.colors', 'message': "Invalid values: ['olive green']"}] | MetadataValidator | None |
| 10_3_back.jpg | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}, {'field': 'style.season', 'message': 'Value not allowed: fall'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: olive green'}] | MetadataValidator | None |
| 11_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: crop top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | MetadataValidator | None |
| 12_7_additional.jpg | FAILED: [{'field': 'shape_and_fit.sleeves', 'message': 'Value not allowed: none'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 15_1_front.jpg | FAILED: [{'field': 'visual_attributes.colors', 'message': 'Value not allowed: sky_blue'}] | MetadataValidator | None |
| 20_4_full.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 21_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'visual_attributes.colors', 'message': 'Value not allowed: teal'}] | MetadataValidator | None |
| 22_7_additional.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | MetadataValidator | None |
| 30_3_back.jpg | FAILED: [{'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: square'}] | MetadataValidator | None |
| 31_2_side.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 49_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}] | MetadataValidator | None |
| 52_1_front.jpg | FAILED: [{'field': 'classification.subcategory', 'message': 'Value not allowed: tank top'}, {'field': 'shape_and_fit.neckline', 'message': 'Value not allowed: v-neck'}] | MetadataValidator | None |
