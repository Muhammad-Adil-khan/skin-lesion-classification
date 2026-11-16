"""Evaluation metrics, written with plain NumPy so they are easy to read and test.

Conventions: `probs` is an (N, K) array of model probabilities, `true_idx` and `pred_idx` are
integer class indices (0..K-1) in the model's output order.
"""
from itertools import permutations

import numpy as np


def confusion_matrix(true_idx: np.ndarray, pred_idx: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_idx, pred_idx):
        matrix[true, pred] += 1
    return matrix


def accuracy(matrix: np.ndarray) -> float:
    return float(np.trace(matrix) / matrix.sum()) if matrix.sum() else float("nan")


def per_class_report(matrix: np.ndarray, class_names: list[str]) -> list[dict]:
    """Precision, recall and F1 for every class. Recall = "of the real X images, how many did we find"."""
    rows = []
    for i, name in enumerate(class_names):
        true_positive = matrix[i, i]
        support = matrix[i, :].sum()  # real images of this class
        predicted = matrix[:, i].sum()  # images the model called this class
        precision = true_positive / predicted if predicted else float("nan")
        recall = true_positive / support if support else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
        rows.append({"class": name, "precision": float(precision), "recall": float(recall),
                     "f1": float(f1), "support": int(support)})
    return rows


def balanced_accuracy(matrix: np.ndarray) -> float:
    """Average recall over classes that appear. Fairer than accuracy when classes are imbalanced."""
    recalls = [row["recall"] for row in per_class_report(matrix, [str(i) for i in range(len(matrix))])
               if row["support"] > 0]
    return float(np.mean(recalls)) if recalls else float("nan")


def one_vs_rest_sensitivity_specificity(matrix: np.ndarray, class_index: int) -> tuple[float, float]:
    """Sensitivity (recall) and specificity for one class against all the others."""
    true_positive = matrix[class_index, class_index]
    false_negative = matrix[class_index, :].sum() - true_positive
    false_positive = matrix[:, class_index].sum() - true_positive
    true_negative = matrix.sum() - true_positive - false_negative - false_positive
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else float("nan")
    return float(sensitivity), float(specificity)


def expected_calibration_error(probs: np.ndarray, true_idx: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """How far the model's confidence is from its real accuracy (0 = perfectly calibrated).

    Predictions are grouped by confidence; in each group we compare average confidence with
    the fraction that were actually right. ECE is the weighted average of those gaps.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += in_bin.mean() * gap
        table.append({"range": f"{low:.1f}-{high:.1f}", "count": int(in_bin.sum()),
                      "mean_confidence": float(confidence[in_bin].mean()),
                      "accuracy": float(correct[in_bin].mean())})
    return float(ece), table


def coverage_accuracy_table(probs: np.ndarray, true_idx: np.ndarray,
                            thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """If we only trusted predictions above each confidence threshold, how many remain and how accurate are they?"""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == true_idx
    rows = []
    for threshold in thresholds:
        kept = confidence >= threshold
        rows.append({"threshold": float(threshold), "coverage": float(kept.mean()),
                     "accuracy": float(correct[kept].mean()) if kept.any() else float("nan")})
    return rows


def suggest_min_confidence(probs: np.ndarray, true_idx: np.ndarray, target_accuracy: float = 0.9,
                           min_coverage: float = 0.2) -> float | None:
    """Lowest confidence threshold whose kept predictions reach `target_accuracy`.

    Returns None if no threshold does that while still keeping at least `min_coverage` of the images.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == true_idx
    for threshold in np.round(np.arange(0.30, 1.0, 0.01), 2):
        kept = confidence >= threshold
        if kept.mean() < min_coverage:
            break
        if correct[kept].mean() >= target_accuracy:
            return float(threshold)
    return None


def class_order_check(probs: np.ndarray, true_names: list[str], class_names: list[str]) -> dict:
    """Test whether the class order in class_names.json really matches the model's outputs.

    We try every possible assignment of labels to output positions and see which one gives the
    best accuracy. If the current order is right, it should win by a wide margin.
    """
    true_names = np.asarray([name.upper() for name in true_names])
    predicted_positions = probs.argmax(axis=1)

    def accuracy_for(order: tuple[str, ...]) -> float:
        return float(np.mean(np.asarray(order)[predicted_positions] == true_names))

    current = tuple(name.upper() for name in class_names)
    ranking = sorted(((accuracy_for(order), order) for order in permutations(current)), reverse=True)
    best_accuracy, best_order = ranking[0]
    runner_up = ranking[1][0]
    return {
        "current_order": list(current),
        "current_accuracy": accuracy_for(current),
        "best_order": list(best_order),
        "best_accuracy": best_accuracy,
        "runner_up_accuracy": runner_up,
        "current_order_is_best": best_order == current,
    }
