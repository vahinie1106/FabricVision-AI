import uvicorn

if __name__ == "__main__":
    print("Starting FabricVision-AI FastAPI Backend...")
    uvicorn.run("backend_api.main:app", host="0.0.0.0", port=8000, reload=True)
