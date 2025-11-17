import numpy as np
from PIL import Image

from dermai.params import MODEL_DATA

def preprocess_image(img: Image.Image, model) -> np.ndarray:
    """Convert a PIL image into a tensor ready for DenseNet."""
    img = img.convert("RGB")
    img = img.resize(MODEL_DATA[model.name]["image_dim"])
    x = np.array(img, dtype=np.float32)
    x = MODEL_DATA[model.name]["preprocess_method"](x) # model builtin preprocessing
    x = np.expand_dims(x, axis=0) # batch of size 1

    return x
