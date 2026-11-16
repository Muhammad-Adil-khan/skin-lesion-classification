"""Tests for the evaluation toolkit, using small synthetic examples with known answers."""
import csv
import sys

import numpy as np
import pytest
from PIL import Image

from app.main import app
from evaluation import metrics, run_evaluation

NAMES = ["BCC", "BKL", "MEL", "NV"]


# ---------- basic metrics ----------
def test_confusion_matrix_and_accuracy():
    true = np.array([0, 0, 1, 1, 2, 3])
    pred = np.array([0, 1, 1, 1, 2, 0])
    matrix = metrics.confusion_matrix(true, pred, 4)
    assert matrix.tolist() == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    assert metrics.accuracy(matrix) == pytest.approx(4 / 6)


def test_per_class_precision_recall_f1():
    matrix = np.array([[8, 2], [4, 6]])  # class 0: 10 real images, 12 predicted
    first, second = metrics.per_class_report(matrix, ["A", "B"])
    assert first["recall"] == pytest.approx(0.8) and first["precision"] == pytest.approx(8 / 12)
    assert first["support"] == 10
    assert second["recall"] == pytest.approx(0.6)
    assert first["f1"] == pytest.approx(2 * (8 / 12) * 0.8 / ((8 / 12) + 0.8))


def test_balanced_accuracy_is_fairer_than_accuracy_on_imbalanced_data():
    matrix = np.array([[95, 5], [8, 2]])  # 97% of "A" right but only 20% of "B"
    assert metrics.accuracy(matrix) == pytest.approx(97 / 110)
    assert metrics.balanced_accuracy(matrix) == pytest.approx((0.95 + 0.2) / 2)


def test_one_vs_rest_sensitivity_and_specificity():
    matrix = np.array([[50, 5, 0], [10, 30, 5], [0, 0, 100]])  # class 1 = "melanoma"
    sensitivity, specificity = metrics.one_vs_rest_sensitivity_specificity(matrix, 1)
    assert sensitivity == pytest.approx(30 / 45)
    assert specificity == pytest.approx(150 / (150 + 5))


def test_calibration_error_is_zero_for_a_calibrated_model():
    rng = np.random.default_rng(0)
    n = 20000
    confidence = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < confidence  # right exactly as often as it claims
    probs = np.zeros((n, 2))
    probs[:, 0] = confidence
    probs[:, 1] = 1 - confidence
    true = np.where(correct, 0, 1)
    ece, _ = metrics.expected_calibration_error(probs, true)
    assert ece < 0.02


def test_calibration_error_is_large_for_an_overconfident_model():
    probs = np.tile([0.99, 0.01], (1000, 1))
    true = np.array([0] * 500 + [1] * 500)  # only 50% right while claiming 99%
    ece, _ = metrics.expected_calibration_error(probs, true)
    assert ece == pytest.approx(0.49, abs=0.01)


def test_coverage_accuracy_table_and_threshold_suggestion():
    # Confident predictions are right, unsure ones are wrong
    probs = np.array([[0.95, 0.05]] * 60 + [[0.55, 0.45]] * 40)
    true = np.array([0] * 60 + [1] * 40)
    table = {row["threshold"]: row for row in metrics.coverage_accuracy_table(probs, true)}
    assert table[0.0]["accuracy"] == pytest.approx(0.6)
    assert table[0.9]["coverage"] == pytest.approx(0.6) and table[0.9]["accuracy"] == pytest.approx(1.0)
    suggestion = metrics.suggest_min_confidence(probs, true, target_accuracy=0.95)
    assert 0.55 < suggestion <= 0.95


def test_threshold_suggestion_is_none_when_the_target_is_unreachable():
    probs = np.tile([0.9, 0.1], (100, 1))
    true = np.array([0] * 50 + [1] * 50)
    assert metrics.suggest_min_confidence(probs, true, target_accuracy=0.95) is None


# ---------- class order check ----------
def make_scores(hidden_order, n=400, seed=0):
    """Fake model outputs where output position i really means hidden_order[i]."""
    rng = np.random.default_rng(seed)
    truth = rng.choice(NAMES, n)
    probs = np.full((n, 4), 0.05)
    for row, label in enumerate(truth):
        probs[row, hidden_order.index(label)] = 0.85
    return probs, list(truth)


def test_class_order_check_passes_when_the_order_is_right():
    probs, truth = make_scores(NAMES)
    result = metrics.class_order_check(probs, truth, NAMES)
    assert result["current_order_is_best"] and result["current_accuracy"] == pytest.approx(1.0)


def test_class_order_check_catches_a_swapped_order():
    hidden = ["BCC", "BKL", "NV", "MEL"]  # the model's last two outputs are really NV, MEL - but class_names says MEL, NV
    probs, truth = make_scores(hidden)
    result = metrics.class_order_check(probs, truth, NAMES)
    assert not result["current_order_is_best"]
    assert result["best_order"] == hidden
    assert result["current_accuracy"] < 0.6 < result["best_accuracy"]


def test_class_order_check_catches_a_fully_scrambled_order():
    hidden = ["NV", "MEL", "BKL", "BCC"]
    probs, truth = make_scores(hidden, seed=1)
    result = metrics.class_order_check(probs, truth, NAMES)
    assert result["best_order"] == hidden and not result["current_order_is_best"]


# ---------- loading labeled data ----------
def write_image(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_folder_layout_is_read_with_upper_case_labels(tmp_path):
    write_image(tmp_path / "mel" / "a.jpg")
    write_image(tmp_path / "nv" / "sub" / "b.png")
    (tmp_path / "nv" / "notes.txt").write_text("not an image")
    samples = run_evaluation.list_images_by_folder(tmp_path)
    assert sorted(label for _, label in samples) == ["MEL", "NV"]


def test_csv_layout_matches_images_by_name(tmp_path):
    write_image(tmp_path / "part1" / "ISIC_1.jpg")
    write_image(tmp_path / "ISIC_2.jpg")
    with open(tmp_path / "meta.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "dx"])
        writer.writerows([["ISIC_1", "mel"], ["ISIC_2", "akiec"], ["ISIC_MISSING", "nv"]])
    samples = run_evaluation.list_images_from_csv(tmp_path, tmp_path / "meta.csv", "image_id", "dx")
    assert sorted(label for _, label in samples) == ["AKIEC", "MEL"]  # missing file ignored


def test_evaluate_builds_a_complete_report():
    probs, truth = make_scores(NAMES)
    report = run_evaluation.evaluate(probs, truth, NAMES)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["melanoma_sensitivity"] == pytest.approx(1.0)
    assert report["class_order_check"]["current_order_is_best"]
    assert {"ece", "coverage_accuracy", "uncertain_flag", "per_class", "confusion_matrix"} <= set(report)


def test_command_line_run_end_to_end(client, tmp_path, monkeypatch, capsys):
    """Runs the real script on a tiny folder of noise images with the already-loaded model."""
    for label in ("bcc", "bkl", "mel", "nv"):
        for number in range(2):
            write_image(tmp_path / "data" / label / f"{number}.jpg", seed=number)
    write_image(tmp_path / "data" / "akiec" / "x.jpg")  # a class the model cannot predict -> skipped

    class LoadedService:  # reuse the app's loaded model instead of loading it a second time
        def __init__(self, *args):
            self._service = app.state.prediction_service

        def load(self):
            pass

        def __getattr__(self, name):
            return getattr(self._service, name)

    monkeypatch.setattr(run_evaluation, "PredictionService", LoadedService)
    monkeypatch.setattr(sys, "argv", ["run_evaluation", str(tmp_path / "data"), "--report", str(tmp_path / "out.json")])
    run_evaluation.main()

    output = capsys.readouterr().out
    assert "Skipped 1 images" in output and "AKIEC" in output
    assert "Images evaluated: 8" in output
    assert "Class-order check" in output and "Confusion matrix" in output
    assert (tmp_path / "out.json").exists()
