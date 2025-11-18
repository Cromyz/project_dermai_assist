import numpy as np
import pandas as pd
from PIL import Image
from keras import Model

from dermai.params import *
from dermai.ml_logic.registry import load_derm_model
from dermai.ml_logic.preprocessor import preprocess_image

def raw_pred(image: Image.Image, model: Model) -> np.ndarray:
    """Return (top_class_name, top_prob, full_dict_class_to_prob)."""
    x = preprocess_image(image, model)

    return model.predict(x)

def pred(image: Image.Image, model: Model) -> pd.DataFrame:
    raw = raw_pred(image, model)

    pred = pd.DataFrame(raw).T
    pred.index = pred.index.map(CODE_TO_CLASS)
    pred.columns = ["Probabilities"]
    pred = pred.sort_values(by="Probabilities", ascending=False)
    pred["color"] = pred.index.map(CLASS_TO_COLOR)
    pred["hexa"] = pred["color"].map(COLOR_TO_HEXA)
    pred.reset_index(inplace=True)

    return pred
