# Data Directory Architecture: Raw Datasets (`data/raw/`)

## Purpose
This directory stores raw, unmodified images and benchmark dataset artifacts ingested into FabricVision-AI.

## Directory Usage
- `data/raw/garments/`: Unprocessed garment source images.
- `data/raw/deepfashion/`: Benchmark DeepFashion dataset files (Category and Attribute Prediction).
- `data/raw/deepfashion2/`: Benchmark DeepFashion2 dataset files (Landmarks & COCO-format JSON).
- `data/raw/fashionpedia/`: Benchmark Fashionpedia dataset files (Ontology & fine-grained attributes).

## Governance Rules
1. **Immutability**: Raw image files in this directory must never be modified in-place by preprocessing or generation routines.
2. **Access**: All loader adapters in `src/data_management/dataset_loader.py` read from this location.
