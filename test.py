import os
import sys
import joblib
import numpy as np
import mlflow
from mlflow.exceptions import MlflowException

def load_model():
    """
    Attempt to load the latest registered model from MLflow Model Registry.
    If unavailable, fall back to the locally saved model.
    """
    MODEL_NAME = "IrisBestModel"
    LOCAL_MODEL_PATH = "/home/sahu_tanmay2104/iris_pipeline/data/artifacts/model_1.joblib"
    MLFLOW_TRACKING_URI = "file:./mlruns"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        print(f"🔍 Attempting to load model '{MODEL_NAME}' from MLflow registry (Production stage)...")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        print("✅ Model loaded from MLflow registry.")
        return model
    except MlflowException as e:
        print(f"⚠️ Could not load model from MLflow registry: {e}")
        print(f"➡️ Falling back to local model at: {LOCAL_MODEL_PATH}")
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                f"❌ No model found locally or in registry. Path checked: {LOCAL_MODEL_PATH}"
            )
        model = joblib.load(LOCAL_MODEL_PATH)
        print("✅ Model loaded successfully from local file.")
        return model


def run_sanity_check():
    """
    Performs basic validation on the trained model.
    Ensures it loads correctly and produces valid predictions.
    """
    # --- Load model (registry or local) ---
    model = load_model()

    # --- Test 1: Prediction functionality ---
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Known Iris-setosa sample
    try:
        prediction = model.predict(sample_input)
        assert 'setosa' in prediction[0].lower(), (
            f"❌ Expected 'setosa', but model predicted '{prediction[0]}'."
        )
        print(f"✅ Prediction test passed: Model predicted '{prediction[0]}' for sample input.")
    except Exception as e:
        raise AssertionError(f"❌ Prediction failed. Error: {e}")

    # --- Test 2: Model stability check ---
    try:
        for _ in range(3):
            assert np.all(model.predict(sample_input) == prediction)
        print("✅ Model produces stable predictions across multiple runs.")
    except Exception as e:
        raise AssertionError(f"❌ Model predictions are inconsistent. Error: {e}")
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(os.path.join(LOG_DIR, "sanity_check.log"), "w") as f:
        f.write("Sanity check passed: Model loaded and prediction successful.\n")
    print("\n🎉 [SUCCESS] All sanity checks passed successfully!")
    return True


if __name__ == "__main__":
    try:
        run_sanity_check()
    except AssertionError as e:
        print(f"\n[FAILURE] Sanity check FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected failure: {e}")
        sys.exit(1)
