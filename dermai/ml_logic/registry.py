import os
import time
from keras.models import Model, load_model
from google.cloud import storage
from colorama import Fore, Style

from dermai.params import *

def load_derm_model(name: str) -> Model:
    print(Fore.BLUE + f"Load latest model `{name}.keras` from GCS..." + Style.RESET_ALL, end=" => ")

    try:
        model = load_model(
            f"gs://{BUCKET_NAME}/models/{name}.keras",
            custom_objects={"preprocess_input": MODEL_DATA[name]["preprocess_method"]},
            safe_mode=False,
        )
        print("✅ Loaded!")

        return model

    except:
        print(f"\n❌ Model not found in GCS bucket {BUCKET_NAME}")

        return None
