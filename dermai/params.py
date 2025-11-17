import os
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.densenet import preprocess_input as preprocess_input_densenet
from keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_efficientnet

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "dermai", "training_outputs")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

MODEL_DATA = {
    "densenet": {
        "label": "DenseNet",
        "preprocess_method": preprocess_input_densenet,
        "image_dim": (224, 224),
    },
    "vgg16": {
        "label": "VGG16",
        "preprocess_method": preprocess_input_vgg16,
        "image_dim": (256, 256),
    },
    "efficientnet": {
        "label": "EfficientNet",
        "preprocess_method": preprocess_input_efficientnet,
        "image_dim": (224, 224),
    },
}

# ----- dataset classes -----
CODE_TO_CLASS = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
CLASS_TO_NAME = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
COLOR_TO_HEXA = {
    "green": "#7EB42E",
    "yellow": "#EEFF34",
    "red": "#FF6F61",
}
CLASS_TO_COLOR = {
    "akiec": "yellow",
    "bcc": "red",
    "bkl": "green",
    "df": "green",
    "mel": "red",
    "nv": "green",
    "vasc": "green",
}
# ---------------------------
