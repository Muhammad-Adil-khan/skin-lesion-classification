"""Tests on REAL labeled images. They skip themselves until you add some (see fixtures/README.md).

These are the tests that tell you whether the model and the class order are actually right.
"""
import os
from pathlib import Path

import pytest

from app.main import app
from evaluation import metrics
from evaluation.run_evaluation import collect_probabilities, list_images_by_folder

LABELED_DIR = Path(__file__).parent / "fixtures" / "labeled"
MIN_ACCURACY = float(os.getenv("SKIN_TEST_MIN_ACCURACY", "0.5"))  # chance level is 0.25 for 4 classes


@pytest.fixture(scope="module")
def results(client, real_model):
    service = app.state.prediction_service
    samples = []
    if LABELED_DIR.exists():
        samples = [(p, label) for p, label in list_images_by_folder(LABELED_DIR) if label in service.class_names]
    if len(samples) < 8 or len({label for _, label in samples}) < 2:
        pytest.skip("add labeled images under tests/fixtures/labeled/<class>/ (see fixtures/README.md)")
    probs, true_names, _ = collect_probabilities(service, samples)
    return probs, true_names, service.class_names


def test_class_order_matches_the_model(results):
    probs, true_names, class_names = results
    check = metrics.class_order_check(probs, true_names, class_names)
    assert check["current_order_is_best"], (
        f"class_names.json order scores {check['current_accuracy']:.0%} but {check['best_order']} "
        f"scores {check['best_accuracy']:.0%} - the class order is probably wrong"
    )


def test_accuracy_is_well_above_chance(results):
    probs, true_names, class_names = results
    accuracy = sum(class_names[i] == t for i, t in zip(probs.argmax(axis=1), true_names)) / len(true_names)
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.0%} is below {MIN_ACCURACY:.0%}"
