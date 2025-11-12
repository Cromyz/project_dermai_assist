import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input


# ---------- GLOBAL CONFIG ----------

MODEL_PATH = Path("models/skin_lesion_densenet_finetuned.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
IMAGE_SIZE = (224, 224)  # adapt if your model expects a different size


# ---------- MODEL & CLASSES LOADING (with cache) ----------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_class_names():
    """
    Read models/class_names.json.
    Handles two common cases:
    - list ["mel", "bcc", ...]
    - dict {"0": "mel", "1": "bcc", ...}
    """
    with open(CLASS_NAMES_PATH, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # sort by key if it's an index-based dict in string form
        try:
            return [data[str(i)] for i in range(len(data))]
        except Exception:
            # fallback: just take values in natural order
            return list(data.values())
    else:
        raise ValueError("Unknown format for class_names.json")


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert a PIL image into a tensor ready for DenseNet."""
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    x = np.array(img, dtype=np.float32)
    x = preprocess_input(x)          # DenseNet preprocessing
    x = np.expand_dims(x, axis=0)    # batch of size 1
    return x


def predict_top_class(image: Image.Image):
    """Return (top_class_name, top_prob, full_dict_class_to_prob)."""
    model = load_model()
    class_names = load_class_names()

    x = preprocess_image(image)
    preds = model.predict(x)  # shape (1, num_classes)
    preds = preds[0]

    # Softmax in case the model output is not already normalized
    probs = tf.nn.softmax(preds).numpy()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    # full dict {class_name: prob}
    all_probs = {
        class_names[i]: float(probs[i]) for i in range(len(class_names))
    }

    return class_names[top_idx], top_prob, all_probs


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="DermAI Assist - Skin Lesion Demo",
    page_icon="🩺",
    layout="centered",
)

st.title("DermAI Assist")
st.caption("Prototype clinical decision support tool for skin lesion analysis (demo).")

st.markdown("---")

st.header("1. Upload a lesion image")

uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG)…",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run analysis"):
        with st.spinner("Running model inference…"):
            try:
                top_class, top_prob, all_probs = predict_top_class(image)
            except Exception as e:
                st.error(f"Error while running the model: {e}")
            else:
                st.markdown("## 2. Analysis result")

                st.subheader("Most probable lesion type")
                st.markdown(
                    f"**{top_class}** with a probability of **{top_prob * 100:.1f}%**"
                )

                # Optional: show all class probabilities
                with st.expander("Show all class probabilities"):
                    st.write(
                        {
                            cls: f"{p * 100:.1f}%"
                            for cls, p in sorted(
                                all_probs.items(), key=lambda x: x[1], reverse=True
                            )
                        }
                    )

                st.info(
                    "⚠️ This is a demonstration tool only. "
                    "It does not replace medical advice or a consultation with a dermatologist."
                )
else:
    st.info("Start by uploading a lesion image to run the analysis.")
