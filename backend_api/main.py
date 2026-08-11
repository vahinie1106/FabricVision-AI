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
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Started {settings.PROJECT_NAME} v{settings.VERSION}")

    # Warm FLUX in THIS process in the background so /health stays responsive while
    # weights initialize. Parent-process prefetch only fills the disk cache; the API
    # child still needs an in-memory pipeline. FluxManager._load_lock serializes
    # warmup vs first Generate so from_pretrained cannot run twice concurrently.
    warmup_flag = os.environ.get("FLUX_WARMUP_ON_STARTUP", "true").strip().lower()
    if warmup_flag not in ("0", "false", "no", "off"):
        from backend_api.services.flux_warmup import warm_flux_in_api_process

        loop = asyncio.get_running_loop()

        async def _bg_flux_warmup() -> None:
            try:
                result = await loop.run_in_executor(None, warm_flux_in_api_process)
                logger.info("FLUX warmup result: %s", result)
                print(f"[startup] FLUX warmup result: {result}", flush=True)
            except Exception as exc:
                # Do not crash the API — Generate will surface MODEL_LOAD_ERROR.
                logger.exception("FLUX warmup failed: %s", exc)
                print(
                    f"[startup] FLUX warmup FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        asyncio.create_task(_bg_flux_warmup())
        logger.info("FLUX warmup scheduled in background")
        print("[startup] FLUX warmup scheduled in background", flush=True)
    else:
        logger.info("FLUX warmup disabled via FLUX_WARMUP_ON_STARTUP")
