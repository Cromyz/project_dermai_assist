from dermai.params import MODEL_DATA
from dermai.interface.main import pred
from dermai.ml_logic.registry import load_derm_model

available_models = []

for model_name in MODEL_DATA.keys():
    model = load_derm_model(model_name)
    model.name = model_name
    available_models.append(model)
