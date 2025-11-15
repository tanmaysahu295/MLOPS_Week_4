import mlflow
import numpy as np
import sys

def run_sanity_check(experiment_name="iris_poisoning_experiment"):
    """
    Loads the latest MLflow model from the experiment and validates 
    it using a known Iris sample.
    """

    # --------------- Load Experiment -------------
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None, f"Experiment '{experiment_name}' not found."

    print(f"📌 Using MLflow Experiment: {experiment_name}")
    
    # --------------- Get Latest Run -------------
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    assert len(runs) > 0, "No MLflow runs found in the experiment."

    latest_run_id = runs.iloc[0].run_id
    print(f"📌 Latest Run ID: {latest_run_id}")
    #print(mlflow.artifacts.list_artifacts(f"runs:/{latest_run_id}"))
    # --------------- Load Model from MLflow -------------
    model_uri = f"runs:/{latest_run_id}/model"
    print(f"📌 Loading model from: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Test 1/3: Model loaded successfully from MLflow.")
    except Exception as e:
        raise AssertionError(f"❌ Failed to load the model: {e}")

    # --------------- Prediction Check -------------
    sample_input = np.array([5.1, 3.5, 1.4, 0.2])   

    try:
        prediction = model.predict(sample_input)[0]
        print(f"📌 Model predicted: {prediction}")

        assert prediction == "versicolor", \
            f"❌ Expected 'setosa', but model predicted '{prediction}'"

        print("✅ Test 2/3: Prediction is correct ('vericolor').")
    except Exception as e:
        raise AssertionError(f"❌ Prediction failed: {e}")

    print("\n🎉 SUCCESS: All MLflow sanity checks passed!")

if __name__ == "__main__":
    try:
        exp = sys.argv[1] if len(sys.argv) > 1 else "iris_poisoning_experiment"

        run_sanity_check(exp)
    except AssertionError as e:
        print(f"\n🔥 FAILURE: {e}")
        sys.exit(1)
