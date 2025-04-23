import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

if os.path.exists("/kaggle"):
    # Running on Kaggle
    DATA_DIR = Path("/kaggle/input/your-dataset-name")
elif os.path.exists("/content"):
    # Running on Google Colab
    DATA_DIR = Path("/content")
else:
    DATA_DIR = Path("data")

AUDIO_PATH = DATA_DIR / "audios"
AUDIO_CACHE = DATA_DIR / "audio_cache"
FEATURES_CACHE = DATA_DIR / "features_cache"
MODELS_DIR = DATA_DIR / "models"

def run_config():
    for folder in [AUDIO_CACHE, FEATURES_CACHE, MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    tqdm.pandas()
