import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import FEATURES_CACHE
from pathlib import Path
from typing import Tuple, Union

# === Feature Extraction ===
class FeatureExtractor:
    def __init__(self) -> None:
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def traditional(self, y: np.ndarray, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
        # Energy & ZCR
        rmse = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)

        return np.concatenate([
            mfcc.mean(axis=1), delta.mean(axis=1), delta2.mean(axis=1),
            chroma.mean(axis=1), contrast.mean(axis=1), tonnetz.mean(axis=1),
            [rmse.mean()], [zcr.mean()]
        ])

    def wav2vec(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        input_values: torch.Tensor = self.wav2vec_proc(y, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            embeddings: torch.Tensor = self.wav2vec_model(input_values).last_hidden_state
        return embeddings.mean(dim=1).squeeze().numpy()

    def extract(self, y: np.ndarray, sr: int = 16000, mode: str = "traditional", n_mfcc: int = 40) -> np.ndarray:
        return self.traditional(y, sr, n_mfcc=n_mfcc) if mode == "traditional" else self.wav2vec(y, sr)

    def cache_features(self, X: np.ndarray, y: np.ndarray, mode: str, force_update: bool = False) -> None:
        X_path = FEATURES_CACHE / f"X_{mode}.npy"
        y_path = FEATURES_CACHE / f"y_{mode}.npy"
        if force_update or not X_path.exists() or not y_path.exists():
            np.save(X_path, X)
            np.save(y_path, y)

    def load_cached_features(self, mode: str) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        X_path = FEATURES_CACHE / f"X_{mode}.npy"
        y_path = FEATURES_CACHE / f"y_{mode}.npy"
        if X_path.exists() and y_path.exists():
            return np.load(X_path), np.load(y_path)
        return None, None
