# Audio Classification

This repository provides a comprehensive solution for audio classification using state-of-the-art machine learning techniques. The project is designed to preprocess audio data, extract meaningful features, train and evaluate multiple models, and perform inference on new audio samples. It supports both traditional and deep learning pipelines, and is fully containerized for reproducibility and ease of deployment.

## Table of Contents

- [Audio Classification](#audio-classification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Running with Python](#running-with-python)
    - [Using Docker](#using-docker)
    - [Using Docker Compose](#using-docker-compose)
  - [Model Training \& Inference](#model-training--inference)
  - [Results](#results)
  - [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
    - [How MLflow is Used](#how-mlflow-is-used)
    - [Starting the MLflow Tracking Server](#starting-the-mlflow-tracking-server)
    - [Accessing the MLflow UI](#accessing-the-mlflow-ui)
    - [Example: Logging a Run](#example-logging-a-run)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

This project focuses on classifying audio files into different categories based on their content. It leverages libraries such as `librosa`, `catboost`, `lightgbm`, `xgboost`, and `scikit-learn` for feature extraction and model training. The workflow is organized into modular Jupyter notebooks and Python scripts, making it easy to experiment, extend, and deploy. This project uses MLflow for experiment tracking, model registry, and reproducibility. MLflow automatically logs parameters, metrics, artifacts, and models during training and evaluation, making it easy to compare different runs and manage your models.

## Features

- **Data Preprocessing:** Robust scripts for loading, cleaning, and preprocessing audio data.
- **Feature Extraction:** Extraction of traditional and advanced audio features using `librosa` and other libraries.
- **Model Training:** Support for multiple models including CatBoost, LightGBM, XGBoost, and SVM.
- **Model Evaluation:** Comprehensive evaluation metrics and visualization tools.
- **Inference Pipeline:** Easy-to-use inference script for batch prediction on new audio files.
- **Experiment Tracking:** Integrated with MLflow for experiment tracking and model registry.
- **Containerization:** Docker support for reproducible environments and deployment.

## Project Structure

```
.
├── 0.Insights.ipynb           # Data exploration and insights
├── 1.Preprocessing.ipynb      # Data preprocessing steps
├── 2.Feature_Extraction.ipynb # Feature extraction pipeline
├── 3.Models.ipynb             # Model training and evaluation
├── app.py                     # Main application script
├── config.py                  # Configuration settings
├── Dockerfile                 # Docker image definition
├── inference.py               # Inference entry point
├── main.ipynb                 # End-to-end pipeline notebook
├── requirements.txt           # Python dependencies
├── requirements_docker.txt    # Docker-specific dependencies
├── utils.py                   # Utility functions
├── data/                      # Audio data directory
├── models/                    # Saved models
├── modules/                   # Custom modules
├── mlruns/                    # MLflow tracking
├── .docker-compose.yaml       # Docker Compose configuration
├── .dockerignore              # Docker ignore file
├── Readme.md                  # Project documentation
└── ...
```

## Setup

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- [Docker](https://www.docker.com/) (optional, for containerized usage)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd Audio-Classification
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your audio dataset:**
   - Place your audio files in the `data/` directory.

## Usage

### Running with Python

To run the pipeline or inference locally:

```bash
python inference.py --team_id 8 --model_path models/your_model.pkl
```

You can adjust the arguments as needed for your use case.

### Using Docker

1. **Build the Docker image:**

   ```bash
   docker build -t audio-infer .
   ```

2. **Run the Docker container:**

   ```bash
   docker run --rm -v "$(pwd)/data/data_20_files:/data" -v "$(pwd)/data/output:/results" audio-infer --team_id 8
   ```

   - Replace `data/data_20_files` with your input directory and `data/output` with your desired output directory.

3. **Run with a specific model:**
   ```bash
   docker run --rm -v "$(pwd)/data/data_20_files:/data" -v "$(pwd)/data/output:/results" audio-infer --team_id 8 --model_path /path/to/your/model
   ```

### Using Docker Compose

1. **Build and run the container:**
   ```bash
   docker-compose -f .docker-compose.yaml up --build
   ```

## Model Training & Inference

- **Training:** Use the provided Jupyter notebooks (`3.Models.ipynb`, `main.ipynb`) to train and evaluate models.
- **Inference:** Use `inference.py` or the Docker container to run inference on new audio files.

## Results

- Results and predictions are saved in the specified output directory.
- Model performance metrics and experiment tracking are available via MLflow in the `mlruns/` directory.

## Experiment Tracking with MLflow

This project uses **MLflow** for experiment tracking, model registry, and reproducibility. MLflow automatically logs parameters, metrics, artifacts, and models during training and evaluation, making it easy to compare different runs and manage your models.

### How MLflow is Used

- **Automatic Logging:** During model training and evaluation, key metrics, parameters, and artifacts (such as trained models) are logged to the mlruns directory.
- **Model Registry:** Trained models can be registered and versioned, allowing for easy deployment and reproducibility.
- **Visualization:** MLflow provides a web UI to visualize experiments, compare runs, and manage models.

### Starting the MLflow Tracking Server

To visualize your experiments and compare runs, you can start the MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

- This command will start the MLflow tracking server and serve the UI at [http://localhost:5000](http://localhost:5000).
- Make sure you run this command from the root directory of your project (where the mlruns folder is located).

### Accessing the MLflow UI

1. Open your browser and go to [http://localhost:5000](http://localhost:5000).
2. You will see a dashboard where you can:
   - Browse all your experiment runs.
   - Compare metrics and parameters across runs.
   - Download artifacts and models.
   - Register and manage model versions.

### Example: Logging a Run

When you train a model using the provided scripts or notebooks, MLflow will automatically log the run. You can then view all details in the MLflow UI.

---

**Tip:**  
If you are running in a Docker container, you may need to map the port (e.g., `-p 5000:5000`) and ensure the mlruns directory is accessible from both your host and the container.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
