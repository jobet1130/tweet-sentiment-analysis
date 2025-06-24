from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict

def evaluate_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {'accuracy': acc, 'f1_score': f1}

def get_confusion_matrix(y_true, y_pred) -> confusion_matrix:
    return confusion_matrix(y_true, y_pred)

def get_classification_report(y_true, y_pred, digits: int = 4) -> str:
    return classification_report(y_true, y_pred, digits=digits)

def plot_confusion_matrix(cm, title: str = '', labels=None, cmap='Blues'):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
