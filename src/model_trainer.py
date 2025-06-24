from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple

def split_data(X, y, test_size: float = 0.2, seed: int = 42) -> Tuple:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def train_logreg(X_train, y_train, save_path: str):
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def train_svm(X_train, y_train, save_path: str):
    model = LinearSVC(C=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def evaluate(model, X_test, y_test) -> Tuple[float, float]:
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)
