from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model

img_size = (80, 80)
CHANNELS = 3

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("simpsons_model.h5")
labels = joblib.load("labels.pkl")

def prepare(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)  
    img = cv.resize(img, img_size)
    img = img.astype("float32") / 255.0          
    img_ready = np.expand_dims(img, axis=(0, -1)) 
    return img_ready

@app.get("/")
def index():
    return {"message": "Simpsons classifier API"}
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
        img_bytes = await file.read()
        img_ready = prepare(img_bytes)
        preds = model.predict(img_ready)
        top_idx = int(np.argmax(preds[0]))
        return {"predicted_label": labels[top_idx]}

