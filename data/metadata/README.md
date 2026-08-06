# Data Directory Architecture: Garment Metadata Store (`data/metadata/`)

## Purpose
Serves as the persistent flat-file metadata storage layer for canonical, validated garment records.

## Naming & Storage Format
- File naming convention: `garment_000001.json`, `garment_000002.json`, etc.
- Synchronized by `src/data_management/metadata_store.py`.
- Enforces Pydantic schema validation (`GarmentMetadata`) before write.

## Standardized JSON Structure Example
```json
{
  "garment_id": "garment_000001",
  "identity": {
    "category": "upper_wear",
    "gender": "women",
    "season": "summer",
    "occasion": "casual"
  },
  "physical": {
    "fabric": "cotton",
    "texture": "smooth",
    "color": ["white"],
    "pattern": "solid"
  },
  "construction": {
    "neckline": "crew",
    "sleeve": "short",
    "silhouette": "regular",
    "fit": "regular"
  },
  "style": {
    "aesthetic": "minimalist",
    "trend": "classic",
    "fashion_category": "basics"
  },
  "created_at": "2026-08-06T22:00:00Z"
}
```
