print("Importing necessary libraries...")
import argparse
import os
from glob import glob
import pandas as pd
from pathlib import Path
import joblib
from modules.preprocessing import AudioPreprocessor
from modules.feature_extraction import FeatureExtractor
from models.xgboost import XGBoostModel
from modules.pipelines import ModelPipeline
from modules.evaluate import PerformanceAnalyzer

print("Loading necessary libraries...")

def run_batch_inference(model, input_folder, output_path, sr=16000, feature_mode="traditional"):
    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()

    files = sorted(glob(os.path.join(input_folder, "*")), key=lambda x: int(Path(x).stem))

    results = []
    X = []
    for file in files:
        # if not file.is_file(): continue
        y = preprocessor.preprocess(preprocessor.load_audio(str(file), sr=sr))
        if y is not None:
            x = extractor.extract(y, sr=sr, mode=feature_mode, n_mfcc=40)
            X.append(x)
            results.append(file)

    pred = model.predict(X)
    df = pd.DataFrame({
        'file': results,
        'pred': pred,
    })
    df.to_csv(output_path, index=False)
    print(f"✅ Batch inference saved to {output_path}")


def main(input_path, model_path, output_path):

    model = ModelPipeline(model=XGBoostModel)
    model.load_model(run_id=model_path, best_metric="weighted avg_f1-score")
    print("✅ Model loaded successfully")

    if not isinstance(model, ModelPipeline):
        raise ValueError("The loaded model is not a valid ModelPipeline instance.")
    run_batch_inference(model, input_path, output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='predictions.csv')
    args = parser.parse_args()
    print(f"Input Path: {args.input_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Path: {args.output_path}")
    main(args.input_path, args.model_path, args.output_path)