import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_pipeline import prepare_data  # noqa: E402


def test_prepare_data():
    """Split cohérent, colonnes inutiles supprimées, cible absente des features."""
    X_train, X_test, y_train, y_test = prepare_data(
        os.path.join(ROOT, "Churn_Modelling.csv")
    )
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "Exited" not in X_train.columns
    assert "Surname" not in X_train.columns
