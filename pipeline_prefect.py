"""Atelier 3 - Pipeline Prefect (MLOps).

Usage :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow fetch
    python pipeline_prefect.py --flow code
    python pipeline_prefect.py --flow train        # alias : entrainement
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow predict
"""

import argparse
import os
import subprocess
import sys

from prefect import flow, get_run_logger, task

# Constantes de main.py (main() ne s'exécute pas à l'import)
from main import DATA_PATH, EXAMPLE_CUSTOMER, MODEL_PATH
from model_pipeline import (
    evaluate_model,
    load_model,
    predict_one,
    prepare_data,
    save_model,
    train_model,
)

# À REMPLACER par l'URL de ton repo GitHub
REPO_URL = "https://github.com/TON_USER/TON_REPO.git"

# On ne scanne QUE nos fichiers (sinon flake8/bandit parcourent le venv)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd):
    """Lance une commande, log la sortie, lève une erreur si code != 0."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")


# ----------------------------- TASKS : git -------------------------------
@task(name="fetch_project")
def fetch_project():
    """Récupère le projet : git pull si on est dans le repo, sinon git clone."""
    if os.path.isdir(".git"):
        run_cmd(["git", "pull", "--ff-only"])
    else:
        run_cmd(["git", "clone", REPO_URL, "project_repo"])


# ----------------------------- TASKS : setup -----------------------------
@task(name="install_dependencies")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


# ----------------------- TASKS : suivi du code ---------------------------
@task(name="format_code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="code_quality")
def code_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=100", *PROJECT_FILES])


@task(name="code_security")
def code_security():
    # -ll : ne remonte que les problèmes de sévérité moyenne ou haute
    run_cmd([sys.executable, "-m", "bandit", "-q", "-ll", *PROJECT_FILES])


@task(name="unit_tests")
def unit_tests():
    run_cmd([sys.executable, "-m", "pytest", "tests/", "-v"])


# ----------------------- TASKS : data / modèle ---------------------------
@task(name="prepare_data")
def task_prepare_data():
    return prepare_data(DATA_PATH)  # X_train, X_test, y_train, y_test


@task(name="train_model")
def task_train_model(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save_model")
def task_save_model(model):
    save_model(model, MODEL_PATH)


@task(name="load_model")
def task_load_model():
    return load_model(MODEL_PATH)


@task(name="evaluate_model")
def task_evaluate_model(model, X_test, y_test):
    accuracy = evaluate_model(model, X_test, y_test)
    get_run_logger().info("Accuracy : %.2f%%", accuracy * 100)
    return accuracy


@task(name="predict")
def task_predict(model):
    result = predict_one(model, EXAMPLE_CUSTOMER)
    get_run_logger().info(
        "Le client va partir (churn)" if result else "Le client reste"
    )
    return result


# -------------------------------- FLOWS ----------------------------------
@flow(name="fetch")
def fetch_flow():
    fetch_project()


@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    X_train, _, y_train, _ = task_prepare_data()
    model = task_train_model(X_train, y_train)
    task_save_model(model)


@flow(name="evaluate")
def evaluate_flow():
    # les données de test sont nécessaires pour évaluer le modèle chargé
    _, X_test, _, y_test = task_prepare_data()
    model = task_load_model()
    task_evaluate_model(model, X_test, y_test)


@flow(name="predict")
def predict_flow():
    model = task_load_model()
    task_predict(model)


@flow(name="all")
def all_flow():
    # 1re étape : récupération du projet depuis le repo git
    fetch_project()
    # étapes code (après installation des dépendances) puis étapes 2-3-4-6
    code_flow()
    X_train, X_test, y_train, y_test = task_prepare_data()
    model = task_train_model(X_train, y_train)
    task_save_model(model)
    task_evaluate_model(model, X_test, y_test)


FLOWS = {
    "all": all_flow,
    "fetch": fetch_flow,
    "code": code_flow,
    "install": install_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
