from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def test_accuracy():
    actual = [0, 1, 2, 1]
    predicted = [0, 1, 2, 0]

    assert accuracy_score(actual, predicted) == 0.75


def test_confusion_matrix_has_seven_classes():
    actual = [0, 1, 2, 3, 4, 5, 6]
    predicted = [0, 1, 2, 3, 4, 5, 6]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(7)),
    )

    assert matrix.shape == (7, 7)


def test_perfect_predictions():
    actual = [0, 1, 2, 3]
    predicted = [0, 1, 2, 3]

    assert accuracy_score(actual, predicted) == 1.0


def test_metrics_are_between_zero_and_one():
    actual = [0, 1, 2, 1]
    predicted = [0, 1, 1, 1]

    for metric in [
        accuracy_score(actual, predicted),
        precision_score(
            actual, predicted, average="weighted", zero_division=0
        ),
        recall_score(
            actual, predicted, average="weighted", zero_division=0
        ),
        f1_score(
            actual, predicted, average="weighted", zero_division=0
        ),
    ]:
        assert 0 <= metric <= 1