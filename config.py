from pathlib import Path
from tqdm import tqdm
import pandas as pd

DATA_DIR = Path("data")
AUDIO_PATH = DATA_DIR / "audios"
AUDIO_CACHE = DATA_DIR / "audio_cache"
FEATURES_CACHE = DATA_DIR / "features_cache"
MODELS_DIR = DATA_DIR / "models"

def run_config():
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tqdm.pandas()
