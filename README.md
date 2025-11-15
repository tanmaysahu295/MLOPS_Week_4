## 1. Objective
The goal of this experiment is to:
- Demonstrate how data poisoning affects model performance
- Compare feature corruption vs label corruption
- Track all runs using MLflow experiments
- Study how accuracy and metrics decay at poison levels: 5%, 10%, 50%
- Understand defenses and how data quantity requirements change when quality is degraded

 ## 2. Project Structure
'''bash
├── train.py                    # Main experiment & poisoning pipeline
├── test.py                     # MLflow-based sanity test (loads latest model)
├── data_iris/
│   └── iris.csv                # Dataset
├── poison_experiments/
│   ├── model_*.joblib          # Models saved locally
│   ├── poison_results_summary.csv
│   └── accuracy_vs_poison.png
├── mlruns/                     # MLflow experiment logs
└── README.md

## 3. Poisoning Types Implemented
# A. Feature-Noise Poisoning
Random numbers replace feature values for a subset of rows.
arduinoCopy codesepal_length → random value within min–max range
sepal_width  → random value within min–max range

Fraction poisoned:
5%, 10%, 50%

# B. Label-Flip Poisoning
Correct labels are replaced with a random incorrect class.
Example:
nginxCopy codesetosa → versicolor
versicolor → virginica

Fraction poisoned:
5%, 10%, 50%

## 4. Training Process
# Each run performs:
 -Optional poisoning of the dataset
 -Train/test split (stratified)
 -Train a DecisionTreeClassifier(max_depth=3)


# Log:
 -Poison type
 -Poison percentage
 -Accuracy, precision, recall, F1
 -Full sklearn model → MLflow (with signature + input example)
 -Joblib model → local artifact

# MLflow Signature Logging
  pythonCopy codesignature = infer_signature(X_train, model.predict(X_train))
  mlflow.sklearn.log_model(
      model,
      name="model",
      input_example=X_train.iloc[:1],
      signature=signature
  )


## 5. How to Run
Install dependencies:
bashCopy codepip install pandas scikit-learn joblib mlflow matplotlib

Run the poisoning experiment:
bashCopy codepython3 train.py

View MLflow UI:
bashCopy codemlflow ui --backend-store-uri mlruns

Open browser to:
cppCopy codehttp://127.0.0.1:5000


## 6. Sanity-Test Model Loading (test.py)
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


## 7. Output Artifacts
📁 poison_experiments/poison_results_summary.csv
Contains:
poison_typepoison_fractionaccuracyprecisionrecallf1
📈 accuracy_vs_poison.png
Graph showing accuracy decay as poisoning increases.

## 8. Expected Observations
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



🛡️
