import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from modules.preprocessing import AudioPreprocessor
from modules.feature_extraction import FeatureExtractor
from models.lightgbm import LightGBMModel
from models.xgboost import XGBoostModel
from modules.pipelines import ModelPipeline
import warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_NAME = {
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
}

# UI Layout
st.set_page_config(page_title="Audio Classification App", layout="centered")
st.title("🎧 Audio Classification")
st.markdown("Upload an `.mp3` or `.wav` file and select a model to get a prediction.")

# File Upload
uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

# Model Selection
selected_model_name = st.selectbox("Select a model", list(MODEL_NAME.keys()))

# Process if file is uploaded
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "input_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        # Preprocess, extract features, predict
        st.info("🔍 Processing audio...")
        try:
            # Initialize pipeline
            preprocessor = AudioPreprocessor()
            extractor = FeatureExtractor()
            model = ModelPipeline(model=MODEL_NAME[selected_model_name])
            model.load_model_from_registry(model_name=selected_model_name)

            # Preprocess and predict
            start_time = time.time()
            y = preprocessor.preprocess(preprocessor.load_audio(audio_path, sr=16000))
            if y is None:
                st.error("Audio preprocessing failed.")
            else:
                x = extractor.extract(y, sr=16000, mode="traditional", n_mfcc=20)
                pred = model.predict([x])[0]
                elapsed = time.time() - start_time

                # Display result
                st.success(f"✅ Predicted Class: `{pred}`")
                st.write(f"Inference time: `{elapsed:.4f}` seconds")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
