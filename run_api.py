import os
import sys

# Before uvicorn imports the app (and torch), reduce CUDA allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Free port 8000 if a leftover FabricVision uvicorn is still bound (WinError 10013).
from backend_api.utils.port_check import free_api_port

_port_status = free_api_port(port=8000)
print(_port_status["message"])
if _port_status["action"] in ("blocked_other", "blocked_unknown", "kill_failed", "still_busy"):
    print(
        "ERROR: Cannot start API on port 8000. "
        "The Next.js frontend expects http://127.0.0.1:8000 — fix the conflict, do not change ports.",
        file=sys.stderr,
    )
    raise SystemExit(1)

import uvicorn

if __name__ == "__main__":
    print("Starting FabricVision-AI FastAPI Backend...")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    uvicorn.run(
        "backend_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["backend_api", "src"],
        reload_excludes=["outputs", "experiments", "models", "data", "*.png", "*.jpg", "*.json"],
    )
