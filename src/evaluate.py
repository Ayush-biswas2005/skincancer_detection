import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from class_names import CLASS_NAMES


def evaluate_model(y_true, y_pred):

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )

    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("results/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":

    # Temporary test data
    y_true = np.array([
        0, 0, 1, 1, 2, 2,
        3, 4, 4, 5, 5, 6
    ])

    y_pred = np.array([
        0, 0, 1, 2, 2, 2,
        3, 4, 0, 5, 5, 6
    ])

    evaluate_model(y_true, y_pred)