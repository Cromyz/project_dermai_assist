import pandas as pd
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from dermai.params import MODEL_DATA
from dermai.interface.main import pred
from dermai.backend import available_models

app = FastAPI()
# app.state.models = available_models ?

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
async def list_model_labels():
    return [v["label"] for v in MODEL_DATA.values()]

@app.post("/predict")
async def predict(
        model_label: str = Form(...),
        img: UploadFile=File(...),
    ):
    contents = await img.read()
    img = Image.open(BytesIO(contents))

    model_index = [d["label"] for d in MODEL_DATA.values()].index(model_label)
    model = available_models[model_index]

    prediction = pred(img, model)
    print(prediction.to_dict("list"))

    return prediction.to_dict("list")

@app.get("/")
async def root():
    return {"message": "App is running"}
