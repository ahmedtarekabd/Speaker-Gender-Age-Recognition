import os
import json
import joblib
import numpy as np

from config import AUDIO_CACHE, FEATURES_CACHE, MODELS_DIR


#* Audio Caching
def cache_audio(data: np.ndarray, filename: str = "default", force_update=False):
    path = AUDIO_CACHE / f"{filename}.npy"
    if force_update or not path.exists():
        np.save(path, data)

def load_cached_audio(filename: str = "default"):
    path = AUDIO_CACHE / f"{filename}.npy"
    return np.load(path) if path.exists() else None


#* Feature Caching
def cache_features(X, y, feature_name: str = "features", label_name: str = "labels", force_update=False):
    X_path = FEATURES_CACHE / f"{feature_name}.npy"
    y_path = FEATURES_CACHE / f"{label_name}.npy"
    if force_update or not X_path.exists() or not y_path.exists():
        np.save(X_path, X)
        np.save(y_path, y)

def load_cached_features(feature_name: str = "features", label_name: str = "labels"):
    X_path = FEATURES_CACHE / f"{feature_name}.npy"
    y_path = FEATURES_CACHE / f"{label_name}.npy"
    if X_path.exists() and y_path.exists():
        return np.load(X_path), np.load(y_path)
    return None, None


#* Model Caching
def cache_model(model, best_params: dict, model_name: str = None, save_option='default', force_update=False):
    model_class = model.__class__.__name__
    model_folder = MODELS_DIR / (model_name or model_class)
    model_folder.mkdir(exist_ok=True)

    model_path = model_folder / ("model.pkl" if save_option == "joblib" else "model.cbm")
    params_path = model_folder / "best_params.json"

    # Save model
    if force_update or not model_path.exists():
        if save_option == "joblib":
            joblib.dump(model, model_path)
        else:
            model.save_model(model_path)

    # Save best params
    if force_update or not params_path.exists():
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

def load_model(model_class, model_name: str = None, save_option='default'):
    model_class_name = model_class.__name__
    model_folder = MODELS_DIR / (model_name or model_class_name)

    model_path = model_folder / ("model.pkl" if save_option == "joblib" else "model.cbm")
    params_path = model_folder / "best_params.json"

    if not model_path.exists() or not params_path.exists():
        return None, None

    with open(params_path, "r") as f:
        best_params = json.load(f)

    if save_option == "joblib":
        model = joblib.load(model_path)
    else:
        model = model_class()
        model.load_model(model_path)

    return model, best_params
