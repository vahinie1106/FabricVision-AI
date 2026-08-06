# Dataset Architecture Review & Production Requirements

**Date**: 2026-08-05  
**Module**: `src/data_management`  
**Status**: Completed  

---

## 1. Executive Summary
This document provides a comprehensive architectural analysis of the dataset and metadata management subsystems in **FabricVision-AI**. As the platform evolves from a validated AI application into an enterprise-scale **Fashion Intelligence Platform**, the underlying data management architecture must support high-throughput dataset ingestion, strict multi-dimensional metadata validation, persistent entity identity management, and future vector indexing without destabilizing existing AI pipelines (Qwen2.5-VL, FLUX, CatVTON).

---

## 2. Current Architecture & Component Inspection

### 2.1 File System & Ingestion Layer (`src/features/dataset/`, `src/preprocessing/`)
- **Current Pattern**: Basic directory traversal scanning local file systems (e.g., `data/raw/garments/`). Files are loaded on-demand using standard image libraries (`PIL.Image`, `cv2`).
- **Limitation**: Ingestion logic is coupled to static local folder structures. There are no standardized adapters to ingest external benchmark datasets (DeepFashion, DeepFashion2, Fashionpedia) or streaming cloud storage buckets.

### 2.2 Semantic Extraction & Metadata Builder (`src/features/semantic_analysis/`)
- **Current Pattern**: Multimodal LLM (`Qwen2.5-VL-3B-Instruct`) extracts unstructured or semi-structured JSON containing garment attributes (`classification`, `physical_attributes`, `visual_attributes`, `shape_and_fit`, `style`, `fabric_behaviour`).
- **Limitation**: The legacy `MetadataValidator` validates fields against generic JSON dictionaries (`metadata_schema.json` and `controlled_vocabularies.json`), but lacks type-safe runtime validation models (such as Pydantic) to guarantee schema compliance across pipeline boundaries.

### 2.3 Configuration Management (`configs/`)
- **Current Pattern**: Static JSON files (`garment_taxonomy.json`, `controlled_vocabularies.json`, `metadata_schema.json`) define allowed values for colors, patterns, materials, and categories.
- **Limitation**: Taxonomy depth is limited. Modern fashion intelligence demands structured identity (gender, season, occasion), physical traits (fabric, texture, color, pattern), construction details (neckline, sleeve, silhouette, fit), and high-level style aesthetics (trend, fashion style).

---

## 3. Key Scalability Gaps & Limitations

| Dimension | Current State | Production Scalability Gap | Impact |
| :--- | :--- | :--- | :--- |
| **Dataset Ingestion** | Local folder traversal | Lack of polymorphic loader adapters for benchmark datasets | Impedes benchmark training & evaluation |
| **Entity Identity** | Path-based file naming | No centralized, persistent sequential ID generation (`garment_000001`) | Prevents relational mapping & vector indexing |
| **Metadata Validation** | Basic dictionary key checks | Missing Pydantic schema validation & controlled vocab sync | Risk of runtime type errors downstream |
| **Knowledge Indexing** | Isolated metadata files | No vector search layer for visual/attribute similarity | Limits retrieval-augmented generation capabilities |

---

## 4. Production Data Architecture Requirements

To achieve a production-grade data foundation layer:

1. **Polymorphic Dataset Adapter Layer (`src/data_management/dataset_loader.py`)**:
   - Implement `BaseDatasetLoader` interface defining standardized contracts (`load_images`, `load_annotations`, `normalize_metadata`, `load_dataset`).
   - Provide concrete adapters for **DeepFashion**, **DeepFashion2**, **Fashionpedia**, and **Local Datasets**.

2. **Strict Pydantic & Controlled Vocabulary Validation (`src/data_management/schemas.py`, `validators.py`)**:
   - Establish strongly-typed Pydantic schemas (`GarmentIdentity`, `PhysicalAttributes`, `ConstructionAttributes`, `StyleAttributes`, `GarmentMetadata`).
   - Guarantee backward compatibility with Qwen2.5-VL outputs while validating against updated controlled vocabularies.

3. **Persistent Metadata Storage (`src/data_management/metadata_store.py`)**:
   - Standardize automated sequential identity assignment (`garment_000001.json`).
   - Store canonical metadata in structured format within `data/metadata/`.

4. **Directory Separation (`data/`)**:
   - Decouple data artifacts into `raw/`, `processed/`, `metadata/`, `annotations/`, and `outputs/`.
