# FabricVision-AI

FabricVision-AI is an end-to-end, production-grade AI system for fashion semantic analysis, garment taxonomy extraction, and virtual try-on workflow orchestration.

---

## 📁 Repository Architecture

The repository adheres to a clean, modular open-source AI project structure:

```
FabricVision-AI/
├── src/                      # Core AI pipeline modules & source code
│   ├── preprocessing/        # Image cleaning, resizing, and normalization
│   ├── dataset_management/   # Dataset indexing, classification & schema metadata
│   ├── semantic_analysis/    # Qwen2.5-VL inference, parsing, normalization & validation
│   ├── virtual_tryon/        # Virtual Try-On pipelines (CatVTON, FLUX integration)
│   ├── models/               # Model loaders & weight management wrappers
│   └── utils/                # Shared utilities & configuration helpers
│
├── configs/                  # Categorized configuration files
│   ├── preprocessing/        # Preprocessing rules & target specs
│   ├── semantic_analysis/    # Schema, controlled vocabularies & synonym mappings
│   ├── models/               # Model device & precision configs
│   └── virtual_tryon/        # Try-on model runtime parameters
│
├── datasets/                 # Datasets directory
│   ├── raw/                  # Unprocessed input garments
│   ├── processed/            # Preprocessed AI-ready garment images
│   ├── validation/           # Ground-truth & validation sample datasets
│   └── fashion_garments/     # Primary dataset taxonomy repository
│
├── outputs/                  # Execution artifacts & outputs
│   ├── metadata/             # Generated JSON metadata
│   ├── organized_dataset/    # Categorized curated dataset outputs
│   ├── preprocessing/        # Intermediate preprocessed images
│   └── generated_images/     # Output virtual try-on renders
│
├── experiments/              # Research & benchmark experiments
│   ├── benchmarks/           # Scale tests (500-image) & benchmark analysis scripts
│   ├── validation/           # Experimental validation suites
│   └── research/debug/       # Diagnostic & temporary research scripts
│
├── scripts/                  # Command-line execution utilities
│   ├── dataset/              # Dataset preparation utilities
│   ├── testing/validation/   # 50-image validation runners & patch utilities
│   ├── benchmarking/         # Benchmark execution helpers
│   └── maintenance/          # Maintenance & structural utilities
│
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests for core modules
│   ├── integration/          # Pipeline end-to-end integration tests
│   └── validation/           # Schema & validator test suites
│
├── reports/                  # Engineering & benchmark documentation
│   ├── benchmarks/           # Scale test analysis & vocabulary reports
│   ├── validation/           # 50-image validation outputs & metrics
│   └── engineering/          # Technical reports & reorganization audit logs
│
├── docs/                     # System architecture & technical documentation
│   ├── Architecture.md       # High-level system design & components
│   ├── Workflow.md           # End-to-end data processing workflow
│   ├── Development.md        # Environment setup & development guidelines
│   └── Future_Ideas.md       # CatVTON & FLUX integration roadmap
│
├── models/                   # Model weight checkpoints
│   ├── qwen/                 # Qwen2.5-VL-3B-Instruct weights
│   ├── flux/                 # FLUX Kontext checkpoints
│   └── catvton/              # CatVTON weights
│
├── app.py                    # Primary Gradio Web UI Application entrypoint
├── README.md                 # Project documentation
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment configuration
└── .gitignore                # Git exclusions
```

---

## 🚀 Getting Started

### Environment Setup

Using Conda / Mamba:
```bash
conda env create -f environment.yml
conda activate fabricvision-ai
```

Using Pip:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Running Unit & Integration Tests

Execute the automated test suite:
```bash
pytest tests/unit/
pytest tests/integration/
```

### Running the Web UI

Launch the interactive Gradio application:
```bash
python app.py
```

---

## 📊 Modules & Execution

1. **Preprocessing Pipeline**: `python -m src.preprocessing.run_preprocessing`
2. **Dataset Management**: `python -m src.dataset_management.dataset_manager`
3. **Semantic Analysis**: `python -m src.semantic_analysis.run_semantic_analysis`
4. **Validation Execution**: `python scripts/testing/validation/run_validation_50.py`

---

## 📜 License & Citation

FabricVision-AI is built for open-source fashion AI research and enterprise virtual try-on workflows.
