"""
main.py
Point d'entrée : exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --step prepare
    python main.py --step train
    python main.py --step evaluate
    python main.py --step predict
    python main.py --step all
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict_one,
    prepare_data,
    save_model,
    train_model,
)

# les arguments CLI
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"

# Client exemple du notebook (CreditScore, Gender, Age, Tenure, Balance,
# NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
EXAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Customer Churn")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "predict", "all"],
        default="all",
        help="étape à exécuter (défaut : all)",
    )
    parser.add_argument("--data", default=DATA_PATH, help="chemin du CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="chemin du modèle .joblib")
    args = parser.parse_args()

    if args.step == "prepare":
        X_train, X_test, _, _ = prepare_data(args.data)
        print(f"Train : {X_train.shape} | Test : {X_test.shape}")
        print(f"Features : {list(X_train.columns)}")

    elif args.step == "train":
        X_train, _, y_train, _ = prepare_data(args.data)
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    elif args.step == "evaluate":
        _, X_test, _, y_test = prepare_data(args.data)
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)

    elif args.step == "predict":
        model = load_model(args.model)
        result = predict_one(model, EXAMPLE_CUSTOMER)
        print("Le client va partir (churn)" if result else "Le client reste")

    else:  # all
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, args.model)


# main() ne s’exécute que si on lance le fichier directement, et pas quand il est importé.

if __name__ == "__main__":
    main()
