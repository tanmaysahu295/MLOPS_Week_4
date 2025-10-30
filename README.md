# Iris Classification Pipeline

This project implements a complete end-to-end Machine Learning pipeline for classifying the Iris flower dataset. It uses DVC for data and pipeline versioning, MLflow for experiment tracking, and GitHub Actions for Continuous Integration.

## Project Structure

```
iris_pipeline/
├── .dvc/
├── .github/workflows/
│   ├── ci.yml
│   └── pythonapp.yml
├── data/
│   ├── artifacts/
│   │   └── model_1.joblib
│   └── data_iris/
│       └── iris.csv
├── logs/
│   └── sanity_check.log
├── mlruns/
├── dvc.yaml
├── params.yaml
├── README.md
├── requirements.txt
├── test.py
└── train.py
```

## File Descriptions

### 🚀 Core Pipeline & Configuration

*   **`dvc.yaml`**: This is the heart of the DVC pipeline. It defines the stages of our ML workflow (`train`, `test`), their dependencies (scripts, data, parameters), and their outputs (model artifacts, logs). This file ensures that the pipeline is reproducible.

*   **`params.yaml`**: A centralized configuration file for all parameters used in the pipeline. This includes data paths, data split ratios, model hyperparameters, and MLflow tracking information. Separating parameters from code allows for easy experimentation without modifying the source code.

*   **`train.py`**: The Python script responsible for the model training stage. It loads data, splits it, trains a model based on the hyperparameters in `params.yaml`, and saves the final model artifact. It also integrates with MLflow to log parameters, metrics, and the model.

*   **`test.py`**: This script performs a sanity check or evaluation on the trained model. It loads the model artifact produced by the `train` stage and runs a simple test, creating a `sanity_check.log` upon successful completion.

### ⚙️ CI/CD

*   **`.github/workflows/ci.yml`**: The main Continuous Integration workflow for GitHub Actions. This workflow automates the process of testing the pipeline on every push or pull request. It checks out the code, sets up Python, installs dependencies, pulls DVC-tracked data, runs the full pipeline with `dvc repro`, and verifies the output.

*   **`.github/workflows/pythonapp.yml`**: An older or alternative CI workflow. It performs a more basic check by installing dependencies and running the `test.py` script directly, without executing the full DVC pipeline.

### 📦 Data and Artifacts

*   **`data/data_iris/iris.csv`**: The raw dataset used for training and testing the model. This file is typically tracked by DVC.

*   **`data/artifacts/model_1.joblib`**: The output of the `train` stage. This is the serialized, trained machine learning model. This artifact is an output of the DVC pipeline and is also tracked by DVC.

*   **`logs/sanity_check.log`**: The output of the `test` stage. The presence and content of this file indicate whether the sanity check on the model passed successfully.

### 📋 Dependencies & Tracking

*   **`requirements.txt`**: A standard Python file that lists all the project dependencies (e.g., `scikit-learn`, `pandas`, `dvc`, `mlflow`). This ensures that anyone can replicate the environment by running `pip install -r requirements.txt`.

*   **`mlruns/`**: The default directory created by MLflow to store experiment tracking data, including parameters, metrics, and artifacts for each run. This directory is typically added to `.gitignore`.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd iris_pipeline
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull DVC-tracked data (if a remote is configured):**
    ```bash
    dvc pull
    ```

4.  **Reproduce the pipeline:**
    ```bash
    dvc repro
    ```

This will execute the stages defined in `dvc.yaml` in the correct order, regenerating the model and log files.