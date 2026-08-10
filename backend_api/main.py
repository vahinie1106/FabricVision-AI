import os

# Must be set before any CUDA context is created in this process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend_api.config.settings import settings
from backend_api.gateway import register_frontend_gateway
from backend_api.routes import health, generation, tryon, semantic

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
)

# Also expose OpenAPI under the versioned API prefix (clients/docs bookmarks).
@app.get(f"{settings.API_V1_STR}/openapi.json", include_in_schema=False)
async def openapi_v1():
    return app.openapi()

# Local Next.js (:3000) + same-host API (:8000). Kaggle Jupyter proxy origins via regex.
# Do NOT use allow_origins=["*"] with allow_credentials=True.
_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
_extra = os.environ.get("FABRICVISION_CORS_ORIGINS", "").strip()
if _extra:
    _CORS_ORIGINS.extend(
        origin.strip() for origin in _extra.split(",") if origin.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.kaggle\.net",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated outputs statically
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# Include routers
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(generation.router, prefix=settings.API_V1_STR, tags=["Generation"])
app.include_router(tryon.router, prefix=settings.API_V1_STR, tags=["Virtual Try-On"])
app.include_router(semantic.router, prefix=settings.API_V1_STR, tags=["Semantic Analysis"])

# Catch-all → Next.js (must be last so /api, /docs, /outputs keep priority)
register_frontend_gateway(app)

@app.on_event("startup")
async def startup_event():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Started {settings.PROJECT_NAME} v{settings.VERSION}")
