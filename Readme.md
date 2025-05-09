# Audio Classification

This repository contains a collection of Jupyter notebooks and Python scripts for audio classification tasks using various machine learning and deep learning techniques. The main focus is on classifying audio files into different categories based on their content.
The project includes the following components:
- **Data Preprocessing**: Scripts for loading and preprocessing audio data, including feature extraction using libraries like `librosa`.
- **Model Training**: Jupyter notebooks for training different models, including traditional machine learning algorithms and deep learning architectures.
- **Model Evaluation**: Scripts for evaluating the performance of trained models using metrics like accuracy, precision, recall, and F1-score.
- **Visualization**: Tools for visualizing audio data and model performance, including confusion matrices and ROC curves.

## How to Use
1. Clone the repository:
   ```bash
   git clone Adsasda
    cd Audio-Classification
    ```
2. Install the required dependencies:
3. ```bash
   pip install -r requirements.txt
   ```
4. Prepare your audio dataset and place it in the `data/` directory.

### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t audio-infer .
   ```
2. Run the Docker container with your audio files mounted:
   ```bash
   docker run --rm -v "$(pwd)/data/data_20_files:/data" -v "$(pwd)/data/output:/results" audio-infer --team_id 8
    ```
3. The results will be saved in the `data/output` directory.
4. You can also run the container with a specific model:
   ```bash
   docker run --rm -v "$(pwd)/data/data_20_files:/data" -v "$(pwd)/data/output:/results" audio-infer --team_id 8 --model_path /path/to/your/model
   ```
### Using Docker Compose
1. Build the Docker image:
   ```bash
   docker-compose -f .docker-compose.yaml up --build
   ```