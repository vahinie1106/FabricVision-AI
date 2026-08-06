# Data Directory Architecture: Processed Datasets (`data/processed/`)

## Purpose
This directory stores standardized, preprocessed garment and human model images ready for AI model inference (FLUX, CatVTON, Qwen2.5-VL).

## Standards
- **Resolution**: Resized to 768x1024 or 512x768 depending on target pipeline requirements.
- **Normalization**: Normalized RGB channels with uniform aspect ratios and white/transparent backgrounds where appropriate.
- **Masking**: Pre-computed garmentagnostic masks and human body densepose masks for Virtual Try-On pipelines.
