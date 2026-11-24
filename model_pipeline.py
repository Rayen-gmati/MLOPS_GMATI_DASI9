"""
model_pipeline.py
Pipeline ML modulaire pour la prédiction du churn clients (Churn_Modelling.csv).
Code extrait du notebook customer_churn.ipynb.
"""

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")  # pas de fenêtre : on sauvegarde les figures en fichier

TARGET = "Exited"
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """
    Charge et prétraite les données.

    Étapes : lecture du CSV, encodage de Gender (LabelEncoder), suppression des
    colonnes inutiles (RowNumber, CustomerId, Surname, Geography), séparation
    features / cible, puis split train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test.
        random_state (int): graine pour un split reproductible.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du fichier.")

    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = df.drop(columns=COLUMNS_TO_DROP)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier.

    Args:
        X_train: features d'entraînement.
        y_train: cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine du modèle.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, plot_path="confusion_matrix.png"):
    """
    Évalue le modèle : accuracy, rapport de classification, matrice de confusion.

    Args:
        model: modèle entraîné.
        X_test: features de test.
        y_test: cible de test.
        plot_path (str | None): où sauvegarder la matrice de confusion (None = pas de figure).

    Returns:
        float: accuracy sur le jeu de test.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion:")
    print(matrix)

    if plot_path:
        ConfusionMatrixDisplay(confusion_matrix=matrix).plot()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"\nFigure sauvegardée : {plot_path}")

    return accuracy


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(path)
    print(f"Modèle chargé : {path}")
    return model


def predict_one(model, customer):
    """
    Prédit le churn d'un client.

    Args:
        model: modèle entraîné.
        customer (list): [CreditScore, Gender, Age, Tenure, Balance,
                          NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

    Returns:
        int: 1 si le client risque de partir, 0 sinon.
    """
    X = pd.DataFrame([customer], columns=model.feature_names_in_)
    return int(model.predict(X)[0])
