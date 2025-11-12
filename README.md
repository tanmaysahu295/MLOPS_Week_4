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

*   **`dvc.yaml`**: Defines the stages of the ML workflow (`train`, `test`) and ensures reproducibility.
*   **`params.yaml`**: Centralized configuration for all pipeline parameters.
*   **`train.py`**: Trains the model and logs metrics via MLflow.
*   **`test.py`**: Performs a sanity check on the trained model and logs the result.

### ⚙️ CI/CD

*   **`.github/workflows/ci.yml`**: Automates the testing pipeline using GitHub Actions.
*   **`.github/workflows/pythonapp.yml`**: Alternate workflow for dependency and test validation.

### 📦 Data and Artifacts

*   **`data/data_iris/iris.csv`**: Raw dataset.
*   **`data/artifacts/model_1.joblib`**: Trained model artifact.
*   **`logs/sanity_check.log`**: Indicates successful model validation.

### 📋 Dependencies & Tracking

*   **`requirements.txt`**: Lists all dependencies (e.g., `scikit-learn`, `pandas`, `dvc`, `mlflow`).
*   **`mlruns/`**: Directory for MLflow experiment tracking.

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

3.  **Pull DVC-tracked data (if configured):**
    ```bash
    dvc pull
    ```

4.  **Reproduce the pipeline:**
    ```bash
    dvc repro
    ```

---

## 🧠 Stress Testing and Auto-Scaling

To ensure the deployed `iris-api` service performs well under varying loads, **stress testing** and **auto-scaling** were configured and validated:

- The application is deployed on **Google Kubernetes Engine (GKE)** using a **LoadBalancer Service** and a **Horizontal Pod Autoscaler (HPA)**.  
- **HPA** automatically scales the number of pods between **1 and 3 replicas** based on CPU usage metrics.
- Stress tests are executed using tools like **`wrk`**, simulating high concurrency (e.g., 1000+ requests).
- During load testing:
  - The **HPA** increases pod count dynamically to handle load.
  - After the load subsides, pods scale back down automatically.
- This setup ensures the API remains **responsive, fault-tolerant, and cost-efficient** during heavy traffic.
