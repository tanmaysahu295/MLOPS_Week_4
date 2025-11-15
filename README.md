 Iris Dataset Poisoning Experiment — MLflow Tracked Pipeline
This project performs a controlled data poisoning experiment on the Iris dataset to measure how ML model performance degrades under various types and severities of poisoning.
Both feature-noise poisoning and label-flip poisoning are introduced at multiple fractions, and the entire training pipeline is tracked using MLflow.

🧪 1. Objective
The goal of this experiment is to:


Demonstrate how data poisoning affects model performance


Compare feature corruption vs label corruption


Track all runs using MLflow experiments


Study how accuracy and metrics decay at poison levels: 5%, 10%, 50%


Understand defenses and how data quantity requirements change when quality is degraded



📂 2. Project Structure
bashCopy code├── train.py                    # Main experiment & poisoning pipeline
├── test.py                     # MLflow-based sanity test (loads latest model)
├── data_iris/
│   └── iris.csv                # Dataset
├── poison_experiments/
│   ├── model_*.joblib          # Models saved locally
│   ├── poison_results_summary.csv
│   └── accuracy_vs_poison.png
├── mlruns/                     # MLflow experiment logs
└── README.md


🧩 3. Poisoning Types Implemented
✔ A. Feature-Noise Poisoning
Random numbers replace feature values for a subset of rows.
arduinoCopy codesepal_length → random value within min–max range
sepal_width  → random value within min–max range

Fraction poisoned:
5%, 10%, 50%

✔ B. Label-Flip Poisoning
Correct labels are replaced with a random incorrect class.
Example:
nginxCopy codesetosa → versicolor
versicolor → virginica

Fraction poisoned:
5%, 10%, 50%

⚙️ 4. Training Process
Each run performs:


Optional poisoning of the dataset


Train/test split (stratified)


Train a DecisionTreeClassifier(max_depth=3)


Log:


Poison type


Poison percentage


Accuracy, precision, recall, F1


Full sklearn model → MLflow (with signature + input example)


Joblib model → local artifact




MLflow Signature Logging
pythonCopy codesignature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=X_train.iloc[:1],
    signature=signature
)


📝 5. How to Run
Install dependencies:
bashCopy codepip install pandas scikit-learn joblib mlflow matplotlib

Run the poisoning experiment:
bashCopy codepython3 train.py

View MLflow UI:
bashCopy codemlflow ui --backend-store-uri mlruns

Open browser to:
cppCopy codehttp://127.0.0.1:5000


🔍 6. Sanity-Test Model Loading (test.py)
This script:


Loads latest MLflow run


Loads model using mlflow.pyfunc.load_model()


Performs a prediction on a known Iris sample


Ensures signature consistency


Run:
bashCopy codepython3 test.py

Example output:
yamlCopy code📌 Using MLflow Experiment: iris_poisoning_experiment
📌 Latest Run ID: xxxxxx
📌 Loading model from mlflow...
✅ Test 1/3: Model loaded
📌 Predicted: setosa
🎉 SUCCESS: All MLflow sanity checks passed!


📊 7. Output Artifacts
📁 poison_experiments/poison_results_summary.csv
Contains:
poison_typepoison_fractionaccuracyprecisionrecallf1
📈 accuracy_vs_poison.png
Graph showing accuracy decay as poisoning increases.

🧠 8. Expected Observations
✔ Feature-Noise Poisoning


At 5%, model accuracy drops slightly


At 10%, noticeable performance degradation


At 50%, model becomes nearly unusable
Feature corruption makes input distribution unstable → unpredictable splits → poor generalization.



✔ Label-Flip Poisoning


Much more harmful than feature noise


Even 5% label flips strongly reduce accuracy


50% flips → model becomes random guesser
Label noise directly disrupts decision boundaries.



🛡️ 9. Mitigation Strategies for Poisoning Attacks
✔ Data-Level Defenses


Outlier detection (Isolation Forest, LOF)


Clustering-based detection of "odd" samples


Statistical tests on feature distributions


Cross-dataset validation


Label verification using majority voting



✔ Model-Level Defenses


Robust loss functions (Huber, Savage, MAE)


Ensemble methods which resist mislabeled points


Differential privacy models (limits per-sample effect)


Trimmed mean or Krum aggregation (for FL settings)



✔ Pipeline-Level Defenses


Automated data profiling in CI/CD


Drift detection between training data versions


Shadow models to compare poisoning impact


Data versioning with DVC or DeltaLake



📈 10. Data Quantity Requirements Under Poisoning
When data quality degrades:
Poison LevelEffectHow Much More Data Needed?5%Model tolerates+30–40% more clean samples10%Decision boundaries shift+1.5–2× dataset size50%Training becomes unreliableData must be cleaned, not increased
Key insight:
Bad data cannot be compensated by more data. Cleaning > collecting.

🧩 11. Conclusion
This experiment clearly shows:


MLflow enables full tracking of poisoning experiments


Even low-level poisoning severely affects classical ML models


Label poisoning is more dangerous than feature poisoning


Data validation, model robustness, and clean pipelines are essential


This repository provides a complete reference implementation of:
✔ Poison data generation
✔ ML training with artifacts
✔ MLflow experiment management
✔ Model validation and CI automation
