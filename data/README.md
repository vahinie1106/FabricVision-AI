# FabricVision-AI Dataset Lifecycle & Storage Specification

This directory manages the end-to-end dataset lifecycle for **FabricVision-AI**, defining clear boundaries between raw ingestion, AI preprocessing, semantic metadata, annotations, and generated outputs.

---

## 🔄 Dataset Lifecycle Flowchart

```text
       +----------------------------+
       |        Raw Datasets        |  (data/raw/)
       |  (Untouched Ingestion)     |
       +--------------+-------------+
                      |
                      v
       +----------------------------+
       |   Image Preprocessing      |  (data/processed/)
       | (768x1024 / Resized / Norm) |
       +--------------+-------------+
                      |
                      v
       +----------------------------+
       |   Semantic Analysis &      |  (data/metadata/ & data/annotations/)
       |   Pydantic Metadata Store  |  (garment_000001.json)
       +--------------+-------------+
                      |
             +--------+--------+
             |                 |
             v                 v
+--------------------+ +--------------------+
|  FLUX Synthesis    | |  CatVTON Try-On    |
| (data/outputs/     | | (data/outputs/     |
|  generated_garments| |  virtual_tryon/)   |
+--------------------+ +--------------------+
```

---

## 📁 Directory Layout

```text
data/
├── README.md                     # Dataset architecture & lifecycle guide
├── raw/                          # Original untouched datasets (Read-only)
│   ├── garments/                 # Fabric and garment images
│   ├── men/                      # Men's clothing datasets
│   └── women/                    # Women's clothing datasets
├── processed/                    # Preprocessed AI-ready images
│   ├── resized/                  # Standardized 768x1024 resolution images
│   ├── normalized/               # RGB normalized color space images
│   └── augmented/                # Data augmentation variants
├── metadata/                     # Persistent JSON metadata records
│   ├── semantic_analysis/        # Qwen2.5-VL extraction outputs
│   ├── annotations/              # Ground truth metadata attributes
│   └── schemas/                  # Validated Pydantic schema instances
├── annotations/                  # Computer Vision ground truth
│   ├── masks/                    # Agnostic garment segmentation masks
│   ├── bounding_boxes/           # Product bounding box coordinates
│   └── labels/                   # Category & keypoint landmark labels
├── outputs/                      # Model generated results
│   ├── generated_garments/       # FLUX synthesis output (images/ & metadata/)
│   ├── virtual_tryon/            # CatVTON try-on output (images/ & metadata/)
│   └── experiments/              # Benchmark run outputs
├── samples/                      # Micro-testing datasets for unit tests
│   ├── test_images/              # Test sample fabric images
│   └── validation/               # Validation benchmarks
└── archive/                      # Historical datasets & legacy batch outputs
```

---

## 🔒 Storage Rules

1. **RAW DATA (`data/raw/`)**: Never modify. Immutable source of truth.
2. **PROCESSED DATA (`data/processed/`)**: Contain only AI-ready transformed images.
3. **METADATA (`data/metadata/`)**: Pydantic schema-validated JSON records (`garment_000001.json`).
4. **OUTPUTS (`data/outputs/`)**: Generated images and execution logs partitioned by pipeline.
