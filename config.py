from pathlib import Path

DATA_DIR = Path("data")
AUDIO_CACHE = DATA_DIR / "audio_cache"
FEATURES_CACHE = DATA_DIR / "features_cache"
MODELS_DIR = DATA_DIR / "models"

def run_config():
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
