from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from dermai.params import MODEL_DATA
from dermai.interface.main import pred, security_check
from dermai.backend import AVAILABLE_MODELS

app = FastAPI()
# app.state.models = AVAILABLE_MODELS ?

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
async def list_model_labels():
    return [v["label"] for v in list(MODEL_DATA.values())[1:]]

@app.post("/control")
async def control(img: UploadFile=File(...)):
    print("Requesting relevance...")

    contents = await img.read()
    img = Image.open(BytesIO(contents))

    allowed = security_check(img)

    return {"security": "passed" if allowed else "failed"}

@app.post("/predict")
async def predict(
        model_label: str = Form(...),
        img: UploadFile=File(...),
    ):
    print("Requesting prediction...")

    contents = await img.read()
    img = Image.open(BytesIO(contents))

    model_index = [d["label"] for d in MODEL_DATA.values()].index(model_label) - 1
    model = AVAILABLE_MODELS[model_index]

    prediction = pred(img, model)

    return prediction.to_dict("list")

@app.get("/")
async def root():
    return {"message": "App is running"}
