from dermai.params import MODEL_DATA
from dermai.ml_logic.registry import load_derm_model

AVAILABLE_MODELS = []
for model_name in MODEL_DATA.keys():
    model = load_derm_model(model_name)
    model.name = model_name
    AVAILABLE_MODELS.append(model)

SECURITY_MODEL = AVAILABLE_MODELS.pop(0)
