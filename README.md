# FabricVision-AI

FabricVision-AI is an AI-based virtual try-on system structured around a modular foundation for dataset preparation, preprocessing, and future model integration. The repository currently includes a Gradio-based UI scaffold, preprocessing modules, and a dedicated dataset management layer; end-to-end FLUX and CatVTON inference remains a future integration milestone.

## Dataset Management Architecture

FabricVision-AI now includes a dedicated Dataset Management Layer that is responsible for organizing, validating, classifying, and documenting the raw dataset before it reaches preprocessing or future AI models.

### What the dataset management layer does

- Scans dataset folders recursively and discovers products and images.
- Builds an index of each product and its associated assets.
- Validates readability, supported formats, dimensions, and folder integrity.
- Classifies garments using a modular rule-based classifier that can later be replaced by a vision-language model.
- Extracts metadata such as gender, material, pattern, and garment type.
- Generates CSV, JSON, and YAML metadata artifacts.
- Reorganizes the dataset into a production-ready garments directory.
- Produces validation, quality, and summary reports in the reports directory.

### Running dataset management

Install dependencies and run:

```bash
python -m src.dataset_management.dataset_manager
```

### Configuration

The dataset management workflow is controlled by [configs/dataset_management.yaml](configs/dataset_management.yaml).

### Reports

Reports are written to [reports](reports).

### Folder structure

The dataset layer creates an AI-ready layout under [datasets/garments](datasets/garments).

## Image Preprocessing Architecture

The preprocessing layer is intentionally model-agnostic. Its purpose is to prepare raw image datasets for future AI services such as FLUX Kontext or CatVTON without coupling those services to the data-loading or transformation logic.

### What the preprocessing layer does

- Discovers images recursively from the raw dataset directory.
- Loads supported image formats such as JPG, JPEG, and PNG.
- Validates image readability, dimensions, and quality.
- Applies configurable cleaning, background handling, resizing, normalization, and augmentation steps.
- Writes processed images to a structured output directory that preserves the input layout.

### Dataset workflow

1. Place raw images under data/raw.
2. Run the preprocessing pipeline.
3. Review generated outputs under data/processed.
4. Use the processed dataset as input for future AI workflows.

### Running preprocessing

Install dependencies and run:

```bash
python src/preprocessing/run_preprocessing.py
```

### Dependencies

The preprocessing stack relies on Pillow, OpenCV, NumPy, Albumentations, PyYAML, tqdm, and scikit-image.
