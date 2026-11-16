# Skin Disease Detection System

> **Medical disclaimer:** educational decision-support prototype. It is **not** a clinically
> validated diagnostic tool. Model confidence is **not** a confirmed medical diagnosis.

A FastAPI backend serves a trained EfficientNetV2-B0 model; a separate Streamlit app is the UI.
Upload a dermoscopic image and get a class (NV, MEL, BCC, BKL), per-class confidence, an
"uncertain result" flag, image-quality warnings and an optional Grad-CAM heatmap.

```
backend/      FastAPI + TensorFlow/Keras (model loaded once at startup)
  app/          the API            evaluation/   scripts to measure the model on labeled images
  models/       model + class names   tests/     automated tests
frontend/     Streamlit (talks to the backend over HTTP only)
docker-compose.yml   run both with one command
```

## Model facts (read from the model file, not assumed)

| | |
|---|---|
| Input | `224 x 224 x 3`, RGB, float values **0-255** |
| Preprocessing | Built into the model (`Rescaling 1/255` + ImageNet `Normalization`). The backend must not scale/normalize again. |
| Output | softmax over 4 classes |
| Class order | `backend/models/class_names.json` -> `["bcc","bkl","mel","nv"]` (index 0..3) |
| Keras | saved with 3.13.2, so `keras==3.13.2` is pinned |
| Model file | `backend/models/best_model.keras` is the trained model, unmodified (originally named `effientnetv2b0.keras`). Point the backend at another file with `SKIN_MODEL_PATH`. |

**Two things are assumed, not proven, until you run the evaluation below:** the class order in
`class_names.json`, and the resize method (`SKIN_RESIZE_METHOD`, default `tf_bilinear`, which is what
Keras `image_dataset_from_directory` uses). Both can be checked from labeled images.

## Run locally

Backend (terminal 1):

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # runtime + test tools (use requirements.txt for runtime only)
uvicorn app.main:app --reload
```

Backend: http://localhost:8000 - Swagger UI: http://localhost:8000/docs (startup takes ~15 s while the model loads and warms up)

Frontend (terminal 2):

```bash
cd frontend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

Frontend: http://localhost:8501

Run `streamlit run` from inside `frontend/`. It reads `frontend/.streamlit/config.toml`, which turns off
Streamlit's first-run email prompt ("Welcome to Streamlit! ... Email:"). That prompt is Streamlit's own
onboarding question, not part of this app. If you ever see it, just press Enter to leave it blank.

**Reproducible installs:** `requirements.txt` pins the main packages; `requirements.lock` pins every
package exactly (Python 3.12, linux x86_64) and is what Docker and CI use.

## Run with Docker

```bash
docker compose up --build        # then open http://localhost:8501
```

Optional settings go in a `.env` file next to `docker-compose.yml` (for example `SKIN_API_KEY=a-long-random-string`).
Both ports are bound to `127.0.0.1` only; see "Security" before exposing them.

> The Docker files have **not been built or run** by their author (no Docker access on the machine they were
> written on). Their ingredients were checked separately, but expect to fix small things on the first build.

## Configuration (environment variables)

Backend (all start with `SKIN_`):

| Variable | Default | Meaning |
|---|---|---|
| `SKIN_MODEL_PATH` | `backend/models/best_model.keras` | model file |
| `SKIN_CLASS_NAMES_PATH` | `backend/models/class_names.json` | class names in output order |
| `SKIN_RESIZE_METHOD` | `tf_bilinear` | `tf_bilinear`, `pil_bilinear` or `pil_nearest`. **Must match training.** |
| `SKIN_MAX_UPLOAD_MB` | `10` | largest accepted file |
| `SKIN_MAX_IMAGE_PIXELS` | `25000000` | rejects huge-dimension images (decompression bombs) |
| `SKIN_MIN_IMAGE_SIDE` | `64` | rejects tiny images |
| `SKIN_MIN_CONFIDENCE` | `0.5` | top probability below this marks the result "uncertain" |
| `SKIN_MIN_MARGIN` | `0.2` | gap between the top two classes below this marks it "uncertain" |
| `SKIN_BLUR_WARNING_THRESHOLD` | `20` | sharpness below this triggers the blur warning |
| `SKIN_DARK_WARNING_THRESHOLD` / `SKIN_BRIGHT_WARNING_THRESHOLD` | `25` / `235` | brightness warnings |
| `SKIN_LOW_CONTRAST_THRESHOLD` | `10` | "almost no detail" warning |
| `SKIN_API_KEY` | *(empty = off)* | if set, `/api` calls must send header `X-API-Key` |
| `SKIN_RATE_LIMIT_PER_MINUTE` | `60` | per client address; `0` = off |
| `SKIN_CORS_ORIGINS` | `http://localhost:8501` | comma-separated allowed origins |

Frontend: `SKIN_API_BASE_URL` (default `http://localhost:8000`) and `SKIN_API_KEY` (only if the backend uses one).
Example: `SKIN_API_BASE_URL=http://192.168.1.10:8000 streamlit run streamlit_app.py`

The quality and uncertainty thresholds are **starting values, not calibrated**. Calibrate them with the evaluation below.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Model status, classes, input size, model SHA-256, whether heatmaps/API key are available |
| POST | `/api/v1/predict` | multipart field `file` (+ optional `?explain=true`) -> prediction JSON |

Response of `POST /api/v1/predict`:

```json
{
  "success": true,
  "predicted_class": "MEL",
  "confidence": 0.94,
  "predictions": {"BCC": 0.03, "BKL": 0.01, "MEL": 0.94, "NV": 0.02},
  "is_uncertain": false,
  "uncertainty_reasons": [],
  "warnings": [],
  "explanation_png_base64": null,
  "model_version": "479cd8731240",
  "disclaimer": "Educational decision-support prototype only. ..."
}
```

`explanation_png_base64` is only filled with `?explain=true` (a Grad-CAM overlay, max 512 px). `model_version` is the
first 12 characters of the model file's SHA-256 so every result can be traced to a model. (Values above are illustrative.)

| Status | Meaning |
|---|---|
| 400 | empty file |
| 401 | missing/wrong API key |
| 413 | file too large |
| 415 | not JPG/JPEG/PNG |
| 422 | corrupted, too small, or too many pixels |
| 429 | rate limit hit (`Retry-After` header says how long to wait) |
| 503 | model not loaded |

### Try it in Swagger UI

1. Open http://localhost:8000/docs.
2. **GET /health** -> *Try it out* -> *Execute*. Expect `"status": "ok"`, `"model_loaded": true`.
3. **POST /api/v1/predict** -> *Try it out* -> choose a `.jpg`/`.jpeg`/`.png` -> optionally set `explain` to `true` -> *Execute*.
4. Error checks: upload a `.txt` (415), a renamed non-image `fake.jpg` (422), a 20x20 image (422).
5. If you set `SKIN_API_KEY`, Swagger UI has no field for the `X-API-Key` header, so use curl instead:
   `curl -H "X-API-Key: <key>" -F "file=@image.jpg;type=image/jpeg" "http://localhost:8000/api/v1/predict?explain=true"`

## Check the model on real images (do this first)

The automated tests only use synthetic images, so **nothing yet proves the predictions are right**. With labeled
images the model has *not* been trained on (your test split):

```bash
cd backend
# folders named bcc/ bkl/ mel/ nv/ :
python -m evaluation.run_evaluation path/to/test_images --compare-resize
# or HAM10000 style (one folder + metadata.csv with image_id,dx):
python -m evaluation.run_evaluation path/to/images --csv metadata.csv --compare-resize
```

It reports accuracy, per-class precision/recall, the confusion matrix, **melanoma sensitivity**, calibration, how
accurate predictions are at each confidence threshold (use that to set `SKIN_MIN_CONFIDENCE`), and:

* **Class-order check** - tries all 24 label orders and tells you if `class_names.json` does not match the model.
* **Resize comparison** (`--compare-resize`) - accuracy for each resize method; set `SKIN_RESIZE_METHOD` to the winner.

To make this a permanent test, drop a few images into `backend/tests/fixtures/labeled/<class>/`
(see the README there); `tests/test_real_images.py` then checks the class order and a minimum accuracy.

## Security and privacy

What is in place:

* **Images are never saved or logged by the app** - they are processed in memory and discarded. (The web framework
  temporarily spools uploads over about 1 MB to the operating system's temp folder while the request is running and
  deletes them when it ends.) Responses are sent with `Cache-Control: no-store`.
* **Limits:** file size (rejected early from the `Content-Length` header, before the upload is parsed), pixel count,
  minimum size, JPG/PNG only, and a per-client **rate limit** (429 + `Retry-After`).
* **Optional API key** (`SKIN_API_KEY`), compared in constant time. Failed attempts count toward the rate limit.
* One prediction runs at a time inside the process (a lock), so concurrent requests queue instead of corrupting each other.

What is **not** in place - needed before exposing this beyond your own machine:

* **HTTPS.** Run both services behind a reverse proxy that terminates TLS, e.g. Caddy:
  ```
  your-domain.example {
      reverse_proxy frontend:8501
  }
  ```
  and do not publish port 8000 publicly. Streamlit has no user login of its own.
* **Per-user limits.** The rate limit counts requests per client address. The only client of the backend is the
  Streamlit server, so all end users share one allowance. Add per-user limits at the proxy if you need them.
* The API key is a shared secret, not user accounts. Keys in the in-memory rate limiter reset when the server restarts.

## Tests

```bash
cd backend  && python -m pytest -v    # uses the real model if present, else a tiny stand-in model
cd frontend && python -m pytest -v    # mocked backend + real UI flows; the live test runs only if a backend is up
```

Backend: 80 tests pass and 2 skip (the real-image tests wait for labeled images). Frontend: 32 pass.
GitHub Actions runs both suites (`.github/workflows/ci.yml`); this workflow has not been run yet because the
project is not in a git repository.

## Known limitations

* **Out-of-scope images are not detected.** A softmax model always answers. Random noise is classified as NV
  with ~80% confidence by this model, and the "uncertain" flag does not catch that. A non-lesion photo can look
  confident. Detecting this properly needs out-of-distribution data to calibrate against.
* The model knows only NV, MEL, BCC and BKL. Other lesion types are forced into one of the four.
* Grad-CAM shows where the model looked; it has not been checked against real lesions and can highlight
  rulers, hair, or borders.
* Quality-warning and uncertainty thresholds are uncalibrated starting values.
