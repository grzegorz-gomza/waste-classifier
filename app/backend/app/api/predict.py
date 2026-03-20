from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from app.services.predictor import Predictor

router = APIRouter()
predictor = Predictor()


@router.post("/dl")
async def predict_dl(run_id: str = Form(...), file: UploadFile = File(...)):
    """
    Predict with a Deep Learning model from a specific MLflow run.
    Accepts an image upload and returns class + confidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        class_name, confidence = predictor.predict_dl(run_id, image)
        return JSONResponse({"run_id": run_id, "model_type": "DL", "class": class_name, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml")
async def predict_ml(run_id: str = Form(...), file: UploadFile = File(...)):
    """
    Predict with an XGBoost model from a specific MLflow run.
    Accepts an image upload and returns class + confidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        class_name, confidence = predictor.predict_ml(run_id, image)
        return JSONResponse({"run_id": run_id, "model_type": "ML", "class": class_name, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
