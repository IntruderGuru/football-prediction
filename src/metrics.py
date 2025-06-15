import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def save_classification_report_txt(y_true, y_pred, path="output/report.txt"):
    """Save the classification report as a plain text file."""
    report = classification_report(y_true, y_pred)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(report)


def save_classification_report_json(y_true, y_pred, path="output/metrics.json"):
    """Save the classification report as a JSON file."""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=4)


def save_confusion_matrix_plot(y_true, y_pred, path="output/confusion_matrix.png"):
    """Save the confusion matrix plot as an image file."""
    cm = confusion_matrix(y_true, y_pred, labels=["H", "D", "A"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["H", "D", "A"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
