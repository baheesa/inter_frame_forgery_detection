"""
Stage IV — Stacking-ensemble forgery classification
===================================================
Paper: "Enhanced inter-frame video forgery detection using convolutional
network and stacking ensemble"
       Multimedia Tools and Applications (2026) 85:497
       https://doi.org/10.1007/s11042-026-21684-x

Classifies each video into one of four labels (paper §2.2.4):
  original | frame-insertion | frame-deletion | frame-duplication

Stacking ensemble
-----------------
Base learners
  - RandomForestClassifier   (100 trees, class_weight=balanced)
  - GradientBoostingClassifier (100 estimators)
  - SVC                      (probability=True, StandardScaler pipeline)
  - KNeighborsClassifier     (StandardScaler pipeline)
Meta-learner
  - LogisticRegression

5-fold StratifiedKFold cross-validation reports mean accuracy / precision /
recall / F1 / MSE; the final model is refit on the full training split.

Pipeline order
--------------
1. annotate.py
2. preprocess_extract.py
3. tcn_feat.py
4. ensemble_class.py         ← you are here

Inputs
------
- Annotation JSON with ``forgery_type`` labels (order must match TCN rows)
- ``*.npy`` TCN embeddings from ``tcn_feat.py``

Outputs
-------
- ``output/stacking_model.pkl``
- ``output/plot_values.pkl`` (CV-averaged metrics)
- Confusion-matrix plots for train / val / test

Note
----
Stage V localisation (wavelet histogram difference ∆w + Otsu thresholding) is
described in the paper (§2.2.5) and is not part of this classification script.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_and_evaluate_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit the stacking ensemble with 5-fold CV and evaluate on held-out sets.

    Returns
    -------
    stacking_model : StackingClassifier
        Final model refit on the full training split.
    avg_metrics : dict
        Mean train / val / test metrics across CV folds.
    test_metrics : dict
        Metrics of the final model on the held-out test set.
    """
    # Base models — complementary inductive biases (trees, kernel, neighbours)
    base_models = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
        ),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        (
            "svm",
            Pipeline(
                [("scaler", StandardScaler()), ("svm", SVC(probability=True))]
            ),
        ),
        (
            "knn",
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
        ),
    ]

    # Meta-classifier blends base-model outputs into the final 4-way decision
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(
        estimators=base_models, final_estimator=meta_model
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        "train_accuracy": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "test_precision": [],
        "train_recall": [],
        "val_recall": [],
        "test_recall": [],
        "train_f1": [],
        "val_f1": [],
        "test_f1": [],
        "train_mse": [],
        "val_mse": [],
        "test_mse": [],
    }

    # Stratified CV on the training partition (paper-style reporting)
    for train_index, val_index in skf.split(X_train, y_train):
        X_tr, X_val_fold = X_train[train_index], X_train[val_index]
        y_tr, y_val_fold = y_train[train_index], y_train[val_index]

        stacking_model.fit(X_tr, y_tr)

        y_tr_pred = stacking_model.predict(X_tr)
        y_val_pred = stacking_model.predict(X_val_fold)
        y_test_pred = stacking_model.predict(X_test)

        metrics["train_accuracy"].append(accuracy_score(y_tr, y_tr_pred))
        metrics["val_accuracy"].append(accuracy_score(y_val_fold, y_val_pred))
        metrics["test_accuracy"].append(accuracy_score(y_test, y_test_pred))

        metrics["train_precision"].append(
            precision_score(y_tr, y_tr_pred, average="weighted", zero_division=1)
        )
        metrics["val_precision"].append(
            precision_score(y_val_fold, y_val_pred, average="weighted", zero_division=1)
        )
        metrics["test_precision"].append(
            precision_score(y_test, y_test_pred, average="weighted", zero_division=1)
        )

        metrics["train_recall"].append(
            recall_score(y_tr, y_tr_pred, average="weighted", zero_division=1)
        )
        metrics["val_recall"].append(
            recall_score(y_val_fold, y_val_pred, average="weighted", zero_division=1)
        )
        metrics["test_recall"].append(
            recall_score(y_test, y_test_pred, average="weighted", zero_division=1)
        )

        metrics["train_f1"].append(f1_score(y_tr, y_tr_pred, average="weighted"))
        metrics["val_f1"].append(f1_score(y_val_fold, y_val_pred, average="weighted"))
        metrics["test_f1"].append(f1_score(y_test, y_test_pred, average="weighted"))

        metrics["train_mse"].append(mean_squared_error(y_tr, y_tr_pred))
        metrics["val_mse"].append(mean_squared_error(y_val_fold, y_val_pred))
        metrics["test_mse"].append(mean_squared_error(y_test, y_test_pred))

    avg_metrics = {key: np.mean(metrics[key]) for key in metrics}

    # Final fit on the full training split; evaluate val + test once
    stacking_model.fit(X_train, y_train)
    y_train_pred = stacking_model.predict(X_train)
    y_val_pred = stacking_model.predict(X_val)
    y_test_pred = stacking_model.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(
            y_test, y_test_pred, average="weighted", zero_division=1
        ),
        "recall": recall_score(
            y_test, y_test_pred, average="weighted", zero_division=1
        ),
        "f1": f1_score(y_test, y_test_pred, average="weighted"),
        "mse": mean_squared_error(y_test, y_test_pred),
    }

    display_confusion_matrix(confusion_matrix(y_train, y_train_pred), "Train")
    display_confusion_matrix(confusion_matrix(y_val, y_val_pred), "Validation")
    display_confusion_matrix(confusion_matrix(y_test, y_test_pred), "Test")

    return stacking_model, avg_metrics, test_metrics


def display_confusion_matrix(cm, dataset_type):
    """Plot a confusion matrix for qualitative inspection of class confusions."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {dataset_type} Set")
    plt.show()


def main():
    """
    Load annotations + TCN features, train the stacking ensemble, save artefacts.

    Update ``annotations_path`` and ``tcn_features_path`` to the files produced
    by earlier stages (order of annotation keys must match TCN row order).
    Default split: 60% train / 20% val / 20% test (0.2 then 0.25 of remainder).
    """
    config = {"split_data": True}

    # --- configure these paths to match your local output directory ---
    annotations_path = os.path.abspath("output/annotations/all_annotate.json")
    tcn_features_path = os.path.abspath("output/all_tcn.npy")

    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        tcn_features = np.load(tcn_features_path, allow_pickle=True)

        # Map string forgery labels → integer class indices
        y = np.array(
            [annotations[video_name]["forgery_type"] for video_name in annotations]
        )
        forgery_types = list(set(y))
        y = np.array([forgery_types.index(label) for label in y])

        # 80/20 then 75/25 of train → overall ≈ 60 / 20 / 20
        X_train, X_test, y_train, y_test = train_test_split(
            tcn_features, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )

        stacking_model, avg_metrics, test_metrics = train_and_evaluate_ensemble(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        model_folder = "output"
        os.makedirs(model_folder, exist_ok=True)

        with open(os.path.join(model_folder, "stacking_model.pkl"), "wb") as f:
            pickle.dump(stacking_model, f)

        with open(os.path.join(model_folder, "plot_values.pkl"), "wb") as f:
            pickle.dump(avg_metrics, f)

        print("Model and metrics saved successfully.")
        print(f"Test metrics: {test_metrics}")
        print(f"CV-averaged metrics: {avg_metrics}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
