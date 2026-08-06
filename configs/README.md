# FabricVision-AI Configuration Architecture

This directory contains modular configuration specifications and taxonomy definitions for **FabricVision-AI**.

---

## 📁 Configuration Structure

```text
configs/
├── README.md                     # Configuration architecture guide
├── global/                       # System-wide global settings
│   ├── device_config.yaml        # PyTorch device, CUDA, & precision settings
│   ├── paths.yaml                # Directory paths & workspace roots
│   └── logging.yaml              # Logging formatters and levels
├── semantic_analysis/            # Qwen2.5-VL Fashion Intelligence settings
│   ├── semantic_analysis_config.yaml # Qwen2.5-VL model parameters
│   ├── metadata_schema.json      # Canonical Pydantic schema keys
│   ├── garment_taxonomy.json     # Master fashion taxonomy hierarchy
│   ├── controlled_vocabularies.json # Enforced vocabulary values
│   └── synonym_mapping.json      # Taxonomy normalization mappings
├── custom_generator/             # FLUX & FLUX Kontext Synthesis settings
│   ├── flux_config.yaml          # FLUX diffusion steps, VRAM offload, 4-bit NF4 settings
│   ├── generation_config.yaml    # Image generation defaults (width, height, guidance)
│   ├── customization_schema.json # User customization form field specs
│   └── garment_templates.json    # Standard prompt template strings
├── virtual_tryon/                # CatVTON Virtual Try-On settings
│   ├── catvton_config.yaml       # CatVTON pipeline & checkpoint settings
│   └── tryon_config.yaml         # Mask auto-segmentation & warp alignment parameters
└── archive/                      # Deprecated or legacy configuration files
```

---

## ⚙️ AI Module Mapping

- **Semantic Analysis (`src/semantic_analysis`)**: Consumes `configs/semantic_analysis/*` for taxonomy enforcement and Pydantic validation.
- **Custom Generator (`src/custom_generator`)**: Consumes `configs/custom_generator/*` for FLUX model loader, quantization, and prompt construction.
- **Virtual Try-On (`src/virtual_tryon`)**: Consumes `configs/virtual_tryon/*` for CatVTON mask alignment and try-on synthesis.
- **Global Services (`backend_api/`, `src/common/`)**: Consumes `configs/global/*` for device selection, logging, and storage paths.
