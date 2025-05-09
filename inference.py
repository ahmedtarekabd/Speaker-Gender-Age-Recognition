import argparse
import os
import time
from glob import glob
import pandas as pd
from pathlib import Path
from modules.preprocessing import AudioPreprocessor
from modules.feature_extraction import FeatureExtractor
from models.lightgbm import LightGBMModel
from models.xgboost import XGBoostModel
from modules.pipelines import ModelPipeline
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = {
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
}

def run_batch_inference(model, input_folder, output_folder, sr=16000, feature_mode="traditional"):
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()

    # Sort files in the correct order
    files = sorted(glob(os.path.join(input_folder, "*")), key=lambda x: int(Path(x).stem))

    results = []
    X = []
    for file in files:
        y = preprocessor.preprocess(preprocessor.load_audio(str(file), sr=sr))
        if y is not None:
            x = extractor.extract(y, sr=sr, mode=feature_mode, n_mfcc=20)
            X.append(x)
            results.append(Path(file).name)

    # Measure inference time
    start_time = time.time()
    pred = model.predict(X)
    end_time = time.time()

    # Save results to results.txt
    results_path = os.path.join(output_folder, "results.txt")
    with open(results_path, "w") as f:
        for file, prediction in zip(results, pred):
            f.write(f"{file},{prediction}\n")
    print(f"✅ Results saved to {results_path}")

    # Save inference time to time.txt
    time_path = os.path.join(output_folder, "time.txt")
    with open(time_path, "w") as f:
        f.write(f"{end_time - start_time:.6f}\n")
    print(f"✅ Inference time saved to {time_path}")

def main(input_path, model_name, output_folder):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    
    if model_name not in MODEL_NAME.keys():
        raise ValueError(f"Model name {model_name} is not valid. Choose from {list(MODEL_NAME.keys())}.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder {output_folder} created.")
    
    model = ModelPipeline(model=MODEL_NAME[model_name])
    model.load_model_from_registry(model_name=model_name)
    print("✅ Model loaded successfully")

    run_batch_inference(model, input_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default="/data", help="Path to the input folder containing test audio files. Default is '/data'.")
    parser.add_argument('--model-name', type=str, default="XGBoost", help="Name of the model to use for inference. Default is 'XGBoost'.")
    parser.add_argument('--team_id', type=str, required=True, help="Team ID for output folder.")
    args = parser.parse_args()

    output_folder = os.path.join("/results", args.team_id)
    print(f"Input Path: {args.input_path}")
    print(f"Model Name: {args.model_name}")
    print(f"Output Folder: {output_folder}")

    main(args.input_path, args.model_name, output_folder)