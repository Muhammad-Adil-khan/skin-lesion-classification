# Labeled test images

`tests/test_real_images.py` runs the model on real images placed here and checks two things:
that `class_names.json` really matches the model's output order, and that accuracy is well above chance.
It is skipped while this folder is empty.

## How to add images

Put dermoscopic images in one sub-folder per class (names are case-insensitive):

```
labeled/
  bcc/  *.jpg
  bkl/  *.jpg
  mel/  *.jpg
  nv/   *.jpg
```

About 20-50 images per class is enough for a smoke test (use the model's **test split**, never images
it was trained on). Keep the images out of public repositories unless the dataset licence allows it.

`SKIN_TEST_MIN_ACCURACY` (default `0.5`) sets the minimum accuracy the test accepts.

For a full evaluation (confusion matrix, melanoma sensitivity, calibration, resize-method comparison) use:

```bash
python -m evaluation.run_evaluation tests/fixtures/labeled --compare-resize
```
