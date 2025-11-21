from tempfile import NamedTemporaryFile
from colorama import Fore, Style
from keras.models import Model, load_model
from google.cloud import storage
from google.api_core.exceptions import NotFound

from dermai.params import *

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def load_derm_model(name: str) -> Model:
    print(Fore.BLUE + f"Load latest model `{name}.keras` from GCS..." + Style.RESET_ALL, end=" => ")

    try:
        blob = bucket.blob(f"models/{name}.keras")
        with NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            blob.download_to_filename(tmp.name)

            # 2) Load the model from the local path (string)
            model = load_model(
                tmp.name,
                custom_objects={"preprocess_input": MODEL_DATA[name]["preprocess_method"]},
                safe_mode=False,
            )
        print("✅ Loaded!")

        return model

    except NotFound:
        print(f"❌ Model `{blob.name}` not found in GCS bucket {BUCKET_NAME}")
        return None

    except Exception as e:
        print(f"❌ Error while loading model `{name}`: {e}")
        return None
