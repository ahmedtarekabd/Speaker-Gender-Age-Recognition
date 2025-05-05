import numpy as np
import librosa
from config import PREPROCESSED_CACHE
import noisereduce as nr
from sklearn.model_selection import train_test_split
from typing import Optional

# === Preprocessing ===
class AudioPreprocessor:
    def load_audio(self, path: str, sr: int = 16000) -> Optional[np.ndarray]:
        try:
            y, _ = librosa.load(path, sr=sr)
            return y
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            return None

    def preprocess(self, y: Optional[np.ndarray], sr: int = 16000, padding: bool = False) -> Optional[np.ndarray]:
        if y is None: return None

        # Remove silence
        intervals = librosa.effects.split(y, top_db=20)
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

        # Normalize volume: Volume variations, Different microphone quality
        y_norm = librosa.util.normalize(y_trimmed)

        # Noise reduction
        y_denoised = nr.reduce_noise(y=y_norm, sr=sr)

        if padding:
            desired_len = sr * 5
            if len(y_denoised) > desired_len:
                y_denoised = y_denoised[:desired_len]
            else:
                y_denoised = np.pad(y_denoised, (0, max(0, desired_len - len(y_denoised))))

        return y_denoised

    def cache_preprocessed(self, idx: str, y: np.ndarray, force_update: bool = False) -> None:
        path = PREPROCESSED_CACHE / f"{idx}.npy"
        if force_update or not path.exists():
            np.save(path, y)

    def load_cached_preprocessed(self, idx: str) -> Optional[np.ndarray]:
        path = PREPROCESSED_CACHE / f"{idx}.npy"
        return np.load(path) if path.exists() else None
    
    def split_data(self, X, y, train_size: float = 0.75, val_size: float = 0.1, random_state: int = 42):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size/(1-train_size), random_state=random_state)
        return X_train, y_train, X_val, y_val, X_test, y_test
