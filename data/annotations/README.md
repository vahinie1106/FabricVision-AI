# Data Directory Architecture: Annotations (`data/annotations/`)

## Purpose
Holds segmentation masks, keypoint annotations, bounding box definitions, and landmark coordinates exported from external fashion benchmark datasets or custom human labeling tools.

## Supported Formats
- **COCO Format**: JSON files formatted for DeepFashion2 and instance segmentation tasks.
- **Fashionpedia Ontology**: COCO-style fine-grained category & attribute mapping arrays.
- **Landmarks**: Coordinate text/JSON maps for collar, hem, sleeve edge, and waist points.
