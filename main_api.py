import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
from typing import List

from yolo_inference import ModelHandler


# --- Pydantic Models ---
class DetectionBox(BaseModel):
    xtl: int
    ytl: int
    xbr: int
    ybr: int


class DetectionResult(BaseModel):
    box: DetectionBox
    confidence: float
    class_id: int
    class_name: str


class InferenceResponse(BaseModel):
    detections: List[DetectionResult]


# --- FastAPI App ---
app = FastAPI(title="YOLOv9 Inference API")

# --- Global Model Handler ---
# Load model at startup using environment variables with defaults
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "weights/best.pt")
DEVICE = os.environ.get("DEVICE", "")  # Autodetect if not set
breakpoint()

print(f"Loading model with weights: {MODEL_WEIGHTS} on device: {DEVICE}")
model_handler = ModelHandler(weights_path=MODEL_WEIGHTS, device=DEVICE)
print("Model loaded successfully.")


@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, performs inference, and returns detected bounding boxes.
    """
    # Read image from upload
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=400, detail="Invalid image file. Could not decode."
            )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Perform inference
    try:
        detections = model_handler.predict(img)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Error during inference.")

    # Format response
    formatted_detections = []
    for det in detections:
        formatted_detections.append(
            DetectionResult(
                box=DetectionBox(
                    xtl=det["box"][0],
                    ytl=det["box"][1],
                    xbr=det["box"][2],
                    ybr=det["box"][3],
                ),
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"],
            )
        )

    return InferenceResponse(detections=formatted_detections)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the YOLOv9 Inference API. Use the /predict endpoint to make predictions."
    }
