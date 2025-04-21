from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
import io
import os
import urllib.request

app = FastAPI()

# Path where model will be saved
model_path = "best.pt"

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

@app.get("/")
def root():
    return {"message": "YOLOv5 API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(image)
    results.render()  # adds boxes
    result_image = Image.fromarray(results.ims[0])

    # Convert image to bytes
    buffer = io.BytesIO()
    result_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
