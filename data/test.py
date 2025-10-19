import joblib
import numpy as np
import os

def run_sanity_check():
    """
    Loads the trained model from disk and performs a simple prediction
    test to ensure it is working correctly.
    """
    MODEL_PATH = "/home/sahu_tanmay2104/iris_pipeline/data/artifacts/model.joblib"
    
    # --- Test 1: Model Existence ---
    # Assert that the model file exists at the specified path.
    # If it doesn't, the script will stop and raise an AssertionError.
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    print("✅ Test 1/3: Model file found successfully.")

    # --- Test 2: Model Loading ---
    # Try to load the model using joblib.
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Test 2/3: Model loaded successfully.")
    except Exception as e:
        # If loading fails for any reason, fail the test.
        assert False, f"Failed to load the model. Error: {e}"

    # --- Test 3: Prediction Check ---
    # This sample is a classic 'setosa' species from the Iris dataset.
    # Features are in the order: ['sepal_length','sepal_width','petal_length','petal_width']
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    try:
        # Get the model's prediction
        prediction = model.predict(sample_input)
        
        # For this specific, well-known sample, the model should predict 'setosa'.
        # This confirms the model is making a reasonable prediction.
        assert prediction[0] == 'setosa', f"Model predicted '{prediction[0]}' for a known 'setosa' sample."
        print(f"✅ Test 3/3: Prediction on sample data is correct ('{prediction[0]}').")

    except Exception as e:
        assert False, f"Prediction failed. Error: {e}"

if __name__ == '_main_':
    try:
        run_sanity_check()
        print("\n[SUCCESS] All sanity checks passed!")
    except AssertionError as e:
        print(f"\n[FAILURE] Sanity check FAILED: {e}")
        # Exit with a non-zero status code to indicate failure, useful for automation
        exit(1)