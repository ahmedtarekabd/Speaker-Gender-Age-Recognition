import numpy as np
import librosa
from config import PREPROCESSED_CACHE
import noisereduce as nr
from sklearn.model_selection import train_test_split
from typing import Optional
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from imblearn.combine import SMOTETomek
import random
from collections import Counter


# === Preprocessing ===
class AudioPreprocessor:
    def __init__(self):
        self.augment_pipeline = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
            PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
            Shift(min_shift=-0.2, max_shift=0.2, p=1.0),
        ])
        self.augment_prob_by_class = {  # set your probabilities here
            0: 0.01,
            1: 0.8,
            2: 0.9,
            3: 0.95
        }

    def load_audio(self, path: str, sr: int = 16000) -> Optional[np.ndarray]:
        try:
            y, _ = librosa.load(path, sr=sr)
            return y
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            return None

    def preprocess(self, y: Optional[np.ndarray], sr: int = 16000, padding: bool = False, label: Optional[int] = None) -> Optional[np.ndarray]:
        if y is None: return None

        # Remove silence
        intervals = librosa.effects.split(y, top_db=20)
        y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

        # Normalize volume: Volume variations, Different microphone quality
        y_norm = librosa.util.normalize(y_trimmed)

        # Noise reduction
        y_denoised = nr.reduce_noise(y=y_norm, sr=sr, n_jobs=-1)


        # Conditional augmentation
        if label is not None and random.random() < self.augment_prob_by_class.get(label, 0.5):
            y_augmented = self.augment_pipeline(samples=y_denoised, sample_rate=sr)
        else:
            y_augmented = y_denoised

        # Padding
        if padding:
            desired_len = sr * 5
            if len(y_augmented) > desired_len:
                y_augmented = y_augmented[:desired_len]
            else:
                y_augmented = np.pad(y_augmented, (0, max(0, desired_len - len(y_augmented))))

        return y_augmented

    def cache_preprocessed(self, idx: str, y: np.ndarray, force_update: bool = False) -> None:
        path = PREPROCESSED_CACHE / f"{idx}.npy"
        if force_update or not path.exists():
            np.save(path, y)

    def load_cached_preprocessed(self, idx: str) -> Optional[np.ndarray]:
        try:
            path = PREPROCESSED_CACHE / f"{idx}.npy"
            return np.load(path) if path.exists() else None
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            return None
    
    def split_data(self, X, y, train_size: float = 0.75, val_size: float = 0.1, random_state: int = 42, stratify: bool = True, 
                    apply_smote: bool = False, smote_percentage: float = 0.7, verbose = True) -> tuple:
        
        # First split: train vs (val + test)
        stratify_option = y if stratify else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=stratify_option
        )

        # Second split: validation vs test
        stratify_temp = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_size / (1 - train_size), random_state=random_state, stratify=stratify_temp
        )

        if apply_smote:
            if verbose: print(f"[INFO] Class distribution before SMOTE: {Counter(y_train)}")
            
            class_counts = Counter(y_train)
            majority_class_count = max(class_counts.values())
            sampling_strategy = {
                cls: int(majority_class_count * smote_percentage) for cls in class_counts.keys()
            }
            sampling_strategy[0] = majority_class_count

            resampler = SMOTETomek(
                random_state=random_state, 
                n_jobs=-1, 
                sampling_strategy=sampling_strategy  # Specify sampling strategy as a dictionary
            )
            X_train, y_train = resampler.fit_resample(X_train, y_train)
            
            if verbose: print(f"[INFO] Class distribution after SMOTE: {Counter(y_train)}")

        return X_train, y_train, X_val, y_val, X_test, y_test
