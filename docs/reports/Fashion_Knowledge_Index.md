# Fashion Knowledge Index Architecture Design

## 1. Vision & Architecture Overview

The **Fashion Knowledge Index (FKI)** is the high-performance vector search and retrieval-augmented layer designed to power next-generation AI capabilities in **FabricVision-AI**. By converting canonical Pydantic garment metadata (`src/data_management/schemas.py`) and preprocessed garment images (`data/processed/`) into multi-modal vector embeddings, FKI enables millisecond-level similarity search, visual garment recommendation, dynamic prompt engineering for FLUX generation, and intelligent garment matching for Virtual Try-On.

```
       +-------------------------------+
       | Canonical Garment Metadata    |
       | & Processed Images            |
       +---------------+---------------+
                       |
                       v
       +-------------------------------+
       | Multi-Modal Embeddings Engine |
       | (OpenCLIP / Qwen2.5-VL / ViT) |
       +---------------+---------------+
                       |
        +--------------+--------------+
        |                             |
        v                             v
+---------------+             +---------------+
| Vector Store  |             | Vector Index  |
|  (ChromaDB)   |             |   (FAISS)     |
+---------------+             +---------------+
        |                             |
        +--------------+--------------+
                       |
                       v
       +-------------------------------+
       |  Fashion Knowledge Index API  |
       +---------------+---------------+
                       |
     +-----------------+-----------------+
     |                                   |
     v                                   v
+-----------------------+     +-----------------------+
| FLUX Garment Design   |     | CatVTON Try-On        |
| Prompt Enrichment     |     | Matching & Retrieval  |
+-----------------------+     +-----------------------+
```

---

## 2. Core Target Technologies

### 2.1 FAISS (Facebook AI Similarity Search)
- **Role**: High-speed, in-memory dense vector indexing and GPU-accelerated similarity search.
- **Index Types**:
  - `IndexFlatIP` (Inner Product / Cosine similarity) for real-time visual vector comparisons.
  - `IndexIVFFlat` (Inverted File Index) for sub-millisecond search across >100,000 garment records.
- **Use Case**: Real-time visual similarity search during user studio uploads and prompt generation.

### 2.2 ChromaDB
- **Role**: Embedded, persistent vector database with structured metadata filtering.
- **Use Case**: Storing vector embeddings alongside canonical Pydantic JSON attributes (`garment_id`, `category`, `fabric`, `season`, `occasion`). Enables hybrid queries like: *"Find garments visually similar to X where fabric = 'silk' AND occasion = 'wedding'"*.

### 2.3 Multi-Modal Embedding Models
- **Visual Embeddings**: `OpenCLIP-ViT-H/14` or `Qwen2.5-VL` dense image feature representations (1024-dim vectors).
- **Textual & Attribute Embeddings**: Text encoders mapping structured garment taxonomies (`garment_taxonomy.json`) into the shared embedding space.

---

## 3. Data Pipeline & Embedding Flow

### 3.1 Indexing Pipeline
1. **Metadata Trigger**: A new garment metadata JSON file is validated and saved by `MetadataStore` (`data/metadata/garment_000001.json`).
2. **Feature Extraction**:
   - The associated processed image (`data/processed/garment_000001.png`) is passed through the vision transformer encoder to produce `V_img`.
   - The canonical metadata dictionary is formatted into a dense textual prompt and passed through the text encoder to produce `V_text`.
   - A composite embedding `V_combined = alpha * V_img + (1 - alpha) * V_text` is generated.
3. **Index Update**:
   - `V_combined` is inserted into FAISS index.
   - Vector + JSON metadata is written to persistent ChromaDB collection `fabricvision_garments`.

### 3.2 Query Engine & Similarity Retrieval
- **Visual Query**: Input garment image -> Vision Encoder -> Top-K Nearest Neighbors in FAISS.
- **Hybrid Query**: Text/Attribute search + Metadata SQL filter -> ChromaDB Vector Query -> Filtered Top-K.

---

## 4. Downstream AI Workflow Integration

### 4.1 FLUX Garment Design Generation
- The Knowledge Index retrieves top matching historical styles and prompts based on user design requests.
- Prompts are dynamically enriched with high-confidence material and silhouette descriptors derived from cluster centroids in the index.

### 4.2 CatVTON Virtual Try-On
- Given a user model image and target garment request, FKI retrieves physically compatible garments with matching drape, thickness, and elasticity profiles to optimize warp map alignment.
