"""
poison_experiment.py

- Generates poisoned datasets (feature-noise or label-flip)
- Trains a DecisionTreeClassifier on each poisoned dataset
- Logs runs to MLflow (params, metrics, model artifact)
- Saves a CSV summary and a matplotlib plot of accuracy vs poison rate

Usage:
    pip install pandas scikit-learn joblib mlflow matplotlib
    python poison_experiment.py
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import mlflow
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = "./data_iris/iris.csv"
OUTPUT_DIR = "./poison_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Poisoning utilities ---
def poison_features_random(df, fraction, features, value_generator=None):
    """
    Replace `fraction` of rows' feature values with random numbers (per-feature).
    - df: pandas DataFrame (copied)
    - fraction: float in (0,1) fraction of rows to poison
    - features: list of column names to poison
    - value_generator: function(feature_name, n) -> ndarray of length n
                       if None, use uniform random in observed feature range
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    k = int(np.floor(fraction * n))
    if k == 0:
        return df

    idx = np.random.choice(n, size=k, replace=False)
    for feat in features:
        if value_generator is None:
            # sample uniform in observed range for that feature
            vmin = df[feat].min()
            vmax = df[feat].max()
            samples = np.random.uniform(low=vmin, high=vmax, size=k)
        else:
            samples = value_generator(feat, k)
        df.loc[idx, feat] = samples
    return df

def poison_labels_flip(df, fraction, label_col='species'):
    """
    Flip `fraction` of labels to a random incorrect class.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    k = int(np.floor(fraction * n))
    if k == 0:
        return df

    classes = df[label_col].unique().tolist()
    idx = np.random.choice(n, size=k, replace=False)
    for i in idx:
        current = df.at[i, label_col]
        choices = [c for c in classes if c != current]
        df.at[i, label_col] = random.choice(choices)
    return df

# --- Training & logging function ---
def train_and_log(df, poison_type, poison_fraction, run_name=None):
    """
    Train DecisionTree on df, evaluate on a fixed clean test set,
    log to MLflow (params, metrics), and save model.
    """
    # We'll split train/test from the dataset each run to simulate the same pipeline:
    train_df, test_df = train_test_split(df, test_size=0.4, stratify=df['species'], random_state=RANDOM_STATE)
    X_train = train_df[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train_df['species']
    X_test = test_df[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test_df['species']

    # Model
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, preds)
    prec = metrics.precision_score(y_test, preds, average='macro', zero_division=0)
    rec = metrics.recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = metrics.f1_score(y_test, preds, average='macro', zero_division=0)
    mlflow.set_experiment("iris_poisoning_experiment")
    # MLflow logging (local file store by default)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poison_type", poison_type)
        mlflow.log_param("poison_fraction", poison_fraction)
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)

        # save model artifact
        model_path = os.path.join(OUTPUT_DIR, f"model_{poison_type}_{int(poison_fraction*100)}.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_path = f"runs:/{mlflow.active_run().info.run_id}/model"

    return {"poison_type": poison_type,
            "poison_fraction": poison_fraction,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "model_path": model_path}


# --- Main experiment loop ---
def main():
    df_orig = pd.read_csv(DATA_PATH)
    features = ['sepal_length','sepal_width','petal_length','petal_width']

    poison_fractions = [0.05, 0.10, 0.50]  # 5%, 10%, 50%
    results = []

    # 0) Baseline (no poisoning)
    r = train_and_log(df_orig, poison_type="none", poison_fraction=0.0, run_name="baseline_clean")
    results.append(r)

    # 1) Feature-noise poisoning (random numbers in feature ranges)
    for frac in poison_fractions:
        df_p = poison_features_random(df_orig, fraction=frac, features=features, value_generator=None)
        run_name = f"feat_noise_{int(frac*100)}pct"
        r = train_and_log(df_p, poison_type="feature_noise", poison_fraction=frac, run_name=run_name)
        results.append(r)

    # 2) Label-flip poisoning (for comparison)
    for frac in poison_fractions:
        df_p = poison_labels_flip(df_orig, fraction=frac, label_col='species')
        run_name = f"label_flip_{int(frac*100)}pct"
        r = train_and_log(df_p, poison_type="label_flip", poison_fraction=frac, run_name=run_name)
        results.append(r)

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUTPUT_DIR, "poison_results_summary.csv")
    results_df.to_csv(results_csv, index=False)
    print("Saved results to", results_csv)

    # Plot accuracy vs poison fraction (features and labels)
    plt.figure()
    for ptype in results_df['poison_type'].unique():
        sub = results_df[results_df['poison_type'] == ptype]
        plt.plot(sub['poison_fraction'], sub['accuracy'], marker='o', label=ptype)
    plt.xlabel("Poison fraction")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Poison fraction")
    plt.legend()
    plot_path = os.path.join(OUTPUT_DIR, "accuracy_vs_poison.png")
    plt.savefig(plot_path)
    print("Saved plot to", plot_path)

if __name__ == "__main__":
    main()
