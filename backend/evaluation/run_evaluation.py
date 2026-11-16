"""Evaluate the trained model on labeled images. Run from the backend/ folder:

  # Images sorted into one sub-folder per class (bcc/, bkl/, mel/, nv/):
  python -m evaluation.run_evaluation path/to/test_images

  # HAM10000-style: one flat folder of images + a metadata CSV (image_id, dx):
  python -m evaluation.run_evaluation path/to/images --csv metadata.csv --image-col image_id --label-col dx

Useful options:
  --compare-resize   try all three resize methods and report accuracy for each (settles which one training used)
  --report out.json  also save the full results as JSON

IMPORTANT: use images the model did NOT see during training (your test split). Images from the
training set give flattering numbers that say nothing about real-world performance.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from app.config import settings
from app.services.prediction_service import PredictionService, assess_uncertainty
from app.utils.preprocessing import InvalidImageError
from evaluation import metrics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
RESIZE_METHODS = ["tf_bilinear", "pil_bilinear", "pil_nearest"]


# ---------------------------------------------------------------- loading the labeled data
def list_images_by_folder(data_dir: Path) -> list[tuple[Path, str]]:
    samples = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((path, class_dir.name.upper()))
    return samples


def list_images_from_csv(data_dir: Path, csv_path: Path, image_col: str, label_col: str) -> list[tuple[Path, str]]:
    # Index every image under data_dir by its file name without the extension
    index = {p.stem: p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    samples = []
    with open(csv_path, newline="") as file:
        for row in csv.DictReader(file):
            path = index.get(row[image_col])
            if path is not None:
                samples.append((path, row[label_col].strip().upper()))
    return samples


# ---------------------------------------------------------------- running the model
def collect_probabilities(service: PredictionService, samples: list[tuple[Path, str]]):
    """Run every image through the model. Returns (probabilities, true_names, number_unreadable)."""
    rows, true_names, unreadable = [], [], 0
    for number, (path, label) in enumerate(samples, start=1):
        try:
            result = service.predict(path.read_bytes())
        except InvalidImageError:
            unreadable += 1
            continue
        rows.append([result["predictions"][name] for name in service.class_names])
        true_names.append(label)
        if number % 100 == 0:
            print(f"  ...{number}/{len(samples)} images", file=sys.stderr)
    return np.asarray(rows), true_names, unreadable


def evaluate(probs: np.ndarray, true_names: list[str], class_names: list[str]) -> dict:
    """Compute every metric. Assumes all labels in `true_names` are model classes."""
    true_idx = np.asarray([class_names.index(name) for name in true_names])
    pred_idx = probs.argmax(axis=1)
    matrix = metrics.confusion_matrix(true_idx, pred_idx, len(class_names))
    ece, calibration = metrics.expected_calibration_error(probs, true_idx)

    flagged = np.asarray([bool(assess_uncertainty(p)) for p in probs])
    correct = pred_idx == true_idx
    report = {
        "images": int(len(true_idx)),
        "accuracy": metrics.accuracy(matrix),
        "balanced_accuracy": metrics.balanced_accuracy(matrix),
        "per_class": metrics.per_class_report(matrix, class_names),
        "confusion_matrix": matrix.tolist(),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "ece": ece,
        "calibration_bins": calibration,
        "coverage_accuracy": metrics.coverage_accuracy_table(probs, true_idx),
        "suggested_min_confidence_for_90pct": metrics.suggest_min_confidence(probs, true_idx, 0.9),
        "uncertain_flag": {
            "settings": {"min_confidence": settings.min_confidence, "min_margin": settings.min_margin},
            "flagged_fraction": float(flagged.mean()),
            "accuracy_when_flagged": float(correct[flagged].mean()) if flagged.any() else None,
            "accuracy_when_not_flagged": float(correct[~flagged].mean()) if (~flagged).any() else None,
        },
        "class_order_check": metrics.class_order_check(probs, true_names, class_names),
    }
    if "MEL" in class_names:
        sensitivity, specificity = metrics.one_vs_rest_sensitivity_specificity(matrix, class_names.index("MEL"))
        report["melanoma_sensitivity"], report["melanoma_specificity"] = sensitivity, specificity
    return report


# ---------------------------------------------------------------- printing
def pct(value) -> str:
    return "n/a" if value is None or value != value else f"{value:.1%}"


def print_report(report: dict, class_names: list[str]) -> None:
    print(f"\nImages evaluated: {report['images']}")
    print(f"Accuracy: {pct(report['accuracy'])}    Balanced accuracy (mean recall): {pct(report['balanced_accuracy'])}")

    print("\nPer class:")
    print(f"  {'class':<6}{'precision':>10}{'recall':>9}{'f1':>8}{'images':>8}")
    for row in report["per_class"]:
        print(f"  {row['class']:<6}{pct(row['precision']):>10}{pct(row['recall']):>9}{pct(row['f1']):>8}{row['support']:>8}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print("        " + "".join(f"{name:>7}" for name in class_names))
    for name, row in zip(class_names, report["confusion_matrix"]):
        print(f"  {name:<6}" + "".join(f"{count:>7}" for count in row))

    if "melanoma_sensitivity" in report:
        print(f"\nMelanoma vs rest: sensitivity {pct(report['melanoma_sensitivity'])} "
              f"(real melanomas caught), specificity {pct(report['melanoma_specificity'])}")

    print(f"\nCalibration: mean confidence {pct(report['mean_confidence'])} vs accuracy {pct(report['accuracy'])}, "
          f"ECE {report['ece']:.3f} (0 = perfect; large = over/under-confident)")
    print("\nIf only predictions at or above a confidence threshold were trusted:")
    for row in report["coverage_accuracy"]:
        print(f"  confidence >= {row['threshold']:.1f}: keeps {pct(row['coverage']):>6} of images, accuracy {pct(row['accuracy'])}")
    suggestion = report["suggested_min_confidence_for_90pct"]
    print("  -> SKIN_MIN_CONFIDENCE for 90% accuracy on kept predictions: "
          + (f"{suggestion:.2f}" if suggestion is not None else "none found (model never reaches 90% while keeping 20% of images)"))

    flag = report["uncertain_flag"]
    print(f"\n'Uncertain' flag with current settings {flag['settings']}: flags {pct(flag['flagged_fraction'])} of images;"
          f" accuracy {pct(flag['accuracy_when_flagged'])} when flagged vs {pct(flag['accuracy_when_not_flagged'])} when not.")

    check = report["class_order_check"]
    print("\nClass-order check:")
    print(f"  current order {check['current_order']}: accuracy {pct(check['current_accuracy'])}")
    if check["current_order_is_best"]:
        print(f"  OK - no other ordering does better (runner-up {pct(check['runner_up_accuracy'])}).")
    else:
        print(f"  WARNING - a different order scores {pct(check['best_accuracy'])}: {check['best_order']}")
        print("  class_names.json may not match the order the model was trained with. Do not use this model until resolved.")


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the skin lesion model on labeled images.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="metadata CSV (use when images are in one flat folder)")
    parser.add_argument("--image-col", default="image_id")
    parser.add_argument("--label-col", default="dx")
    parser.add_argument("--resize-method", choices=RESIZE_METHODS, default=settings.resize_method)
    parser.add_argument("--compare-resize", action="store_true", help="evaluate all three resize methods")
    parser.add_argument("--max-images", type=int, help="only use the first N images (quick check)")
    parser.add_argument("--report", type=Path, help="save the full results to this JSON file")
    args = parser.parse_args()

    service = PredictionService(settings.model_path, settings.class_names_path)
    service.load()

    samples = (list_images_from_csv(args.data_dir, args.csv, args.image_col, args.label_col)
               if args.csv else list_images_by_folder(args.data_dir))
    known = set(service.class_names)
    skipped_labels = sorted({label for _, label in samples if label not in known})
    skipped_count = sum(label not in known for _, label in samples)
    samples = [(path, label) for path, label in samples if label in known]
    if args.max_images:
        samples = samples[:args.max_images]
    if not samples:
        sys.exit(f"No usable images found. The model's classes are {service.class_names}.")
    if skipped_count:
        print(f"Skipped {skipped_count} images with labels the model cannot predict: {skipped_labels}")

    methods = RESIZE_METHODS if args.compare_resize else [args.resize_method]
    all_reports = {}
    for method in methods:
        settings.resize_method = method
        print(f"\n===== resize method: {method} =====")
        probs, true_names, unreadable = collect_probabilities(service, samples)
        if unreadable:
            print(f"({unreadable} unreadable images were skipped)")
        all_reports[method] = evaluate(probs, true_names, service.class_names)
        print_report(all_reports[method], service.class_names)

    if args.compare_resize:
        print("\n===== resize method comparison =====")
        for method, report in all_reports.items():
            print(f"  {method:<13} accuracy {pct(report['accuracy'])}   balanced accuracy {pct(report['balanced_accuracy'])}")
        print("Pick the best one and set SKIN_RESIZE_METHOD to it. A clear winner means it matches training; "
              "near-identical scores mean the choice barely matters.")

    if args.report:
        args.report.write_text(json.dumps(all_reports, indent=2))
        print(f"\nFull results saved to {args.report}")


if __name__ == "__main__":
    main()
