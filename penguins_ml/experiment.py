"""Compare species classifiers without using the held-out set for selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "penguins.csv"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "species"
SEED = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=["NA"])
    required = set(FEATURES + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if frame[TARGET].isna().any():
        raise ValueError("La cible contient des valeurs manquantes")
    return frame


def make_models() -> dict:
    return {
        "Baseline majoritaire": make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="most_frequent")),
        "Régression logistique": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "K plus proches voisins": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Forêt aléatoire": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
    }


def run_experiment(path: Path = DATA_PATH) -> dict:
    frame = load_data(path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.2, stratify=frame[TARGET], random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = make_models()
    comparisons = []
    fitted = {}
    for name, model in models.items():
        scores = cross_validate(
            model, x_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"},
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        comparisons.append({
            "model": name,
            "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
            "cv_balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "test_f1_macro": float(f1_score(y_test, predictions, average="macro")),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
        })
        fitted[name] = model

    # La sélection est faite uniquement sur les plis d'entraînement.
    comparisons.sort(key=lambda item: item["cv_f1_macro_mean"], reverse=True)
    winner = comparisons[0]["model"]
    labels = sorted(frame[TARGET].unique().tolist())
    final_predictions = fitted[winner].predict(x_test)
    return {
        "dataset_rows": len(frame),
        "class_counts": {str(key): int(value) for key, value in frame[TARGET].value_counts().sort_index().items()},
        "missing_features": {key: int(value) for key, value in frame[FEATURES].isna().sum().items()},
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": FEATURES,
        "seed": SEED,
        "cv_folds": 5,
        "selection_metric": "cv_f1_macro_mean",
        "comparisons": comparisons,
        "selected_model": winner,
        "class_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, final_predictions, labels=labels).tolist(),
        "classification_report": classification_report(y_test, final_predictions, labels=labels, output_dict=True, zero_division=0),
    }


def main() -> None:
    results = run_experiment()
    output = ROOT / "results" / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Résultats enregistrés : {output}")
    print(pd.DataFrame(results["comparisons"]).to_string(index=False, float_format=lambda number: f"{number:.3f}"))
    print(f"Modèle sélectionné sur validation croisée : {results['selected_model']}")


if __name__ == "__main__":
    main()
