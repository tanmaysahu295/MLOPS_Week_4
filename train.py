import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlflow.exceptions import MlflowException

warnings.filterwarnings("ignore")

# ----------------------------
# LOAD CONFIG FROM dvc.yaml
# ----------------------------
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_load"]["data_path"]
MODEL_OUTPUT_PATH = config["train"]["model_output_path"]
MLFLOW_TRACKING_URI = config["train"]["tracking_uri"]
EXPERIMENT_NAME = config["train"]["experiment_name"]
REGISTERED_MODEL_NAME = config["train"]["model_name"]
STORAGE_LOCATION = config["train"]["storage_location"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)


def train_and_evaluate():
    """Train multiple models, track with MLflow, and register the best model."""
    # ----------------------------
    # LOAD DATA
    # ----------------------------
    data = load_data(DATA_PATH)
    train, test = train_test_split(
        data, test_size=0.4, stratify=data["species"], random_state=42
    )

    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train.species
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test.species

    # ----------------------------
    # DEFINE MODELS
    # ----------------------------
    models = {
        "DecisionTree": {
            "model": DecisionTreeClassifier,
            "params": [{"max_depth": d} for d in [2, 3, 4, 5]],
        },
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": [
                {"n_estimators": n, "max_depth": d}
                for n in [50, 100]
                for d in [3, 5]
            ],
        },
        "LogisticRegression": {
            "model": LogisticRegression,
            "params": [{"C": c, "max_iter": 200} for c in [0.1, 1.0, 10.0]],
        },
    }

    best_model = None
    best_score = -1
    best_model_name = ""
    best_run_id = None

    # ----------------------------
    # TRAIN MODELS
    # ----------------------------
    for model_name, model_info in models.items():
        for param_set in model_info["params"]:
            with mlflow.start_run(run_name=f"{model_name}_{param_set}"):
                model = model_info["model"](**param_set)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                # Log params & metrics
                mlflow.log_params(param_set)
                mlflow.log_metric("accuracy", acc)

                # Metadata tags
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("data_path", DATA_PATH)

                # Log model
                mlflow.sklearn.log_model(model, model_name)

                print(f"Trained {model_name} with {param_set}, Accuracy={acc:.4f}")

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = model_name
                    best_run_id = mlflow.active_run().info.run_id

    # ----------------------------
    # REGISTER BEST MODEL
    # ----------------------------
    print(f"\n✅ Best model: {best_model_name} (Accuracy={best_score:.4f})")

    model_uri = f"runs:/{best_run_id}/{best_model_name}"

    try:
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        print(f"📦 Registered '{best_model_name}' as '{REGISTERED_MODEL_NAME}' in MLflow registry.")
    except MlflowException as e:
        print(f"⚠️ Model registration skipped: {e}")

    # ----------------------------
    # CONFUSION MATRIX
    # ----------------------------
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.savefig("conf_matrix.png")
    mlflow.log_artifact("conf_matrix.png")
    plt.close()

    # ----------------------------
    # SAVE BEST MODEL LOCALLY
    # ----------------------------
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"💾 Best model saved locally at: {MODEL_OUTPUT_PATH}")
    print(f"🧩 MLflow Artifact Store: {STORAGE_LOCATION}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    train_and_evaluate()
