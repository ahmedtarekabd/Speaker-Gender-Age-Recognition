import numpy as np
import librosa
from config import PREPROCESSED_CACHE

# === Preprocessing ===
class AudioPreprocessor:
    def load_audio(self, path, sr=16000):
        try:
            y, _ = librosa.load(path, sr=sr)
            return y
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            return None

    def preprocess(self, y, sr=16000):
        if y is None:
            return None
        intervals = librosa.effects.split(y, top_db=20)
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
        y_norm = librosa.util.normalize(y_trimmed)
        desired_len = sr * 5
        if len(y_norm) > desired_len:
            y_norm = y_norm[:desired_len]
        else:
            y_norm = np.pad(y_norm, (0, max(0, desired_len - len(y_norm))))
        return y_norm

    def cache_preprocessed(self, idx, y, force_update=False):
        path = PREPROCESSED_CACHE / f"{idx}.npy"
        if force_update or not path.exists():
            np.save(path, y)

    def load_cached_preprocessed(self, idx):
        path = PREPROCESSED_CACHE / f"{idx}.npy"
        return np.load(path) if path.exists() else None
