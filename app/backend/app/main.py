from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import runs, artifacts, predict
from app.core.config import settings

app = FastAPI(
    title="WasteClassifier API",
    description="MLOps web backend for run selection, artifact browsing, and image prediction",
    version="0.1.0",
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
