"""Loads the trained model once and runs predictions with it."""
import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from app.config import settings
from app.utils.explain import heatmap_overlay_base64
from app.utils.preprocessing import prepare_image

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational decision-support prototype only. This is not a clinically validated "
    "diagnostic tool and the confidence score is not a confirmed medical diagnosis. "
    "Consult a qualified dermatologist."
)


class ModelLoadError(Exception):
    """Raised when the model or class names cannot be loaded."""


def file_sha256(path: Path) -> str:
    """Fingerprint of a file, read in chunks so big files do not fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_uncertainty(probabilities: np.ndarray) -> list[str]:
    """Return the reasons the result looks unreliable (an empty list means no concern).

    Note: a softmax model can still be confidently wrong, especially on images unlike its
    training data, so a result that is NOT flagged is not a guarantee.
    """
    ordered = np.sort(probabilities)[::-1]
    top, runner_up = float(ordered[0]), float(ordered[1])
    reasons = []
    if top < settings.min_confidence:
        reasons.append(f"The top confidence ({top:.0%}) is below {settings.min_confidence:.0%}.")
    if top - runner_up < settings.min_margin:
        reasons.append("The two most likely classes are too close together to separate reliably.")
    return reasons


class PredictionService:
    def __init__(self, model_path: Path, class_names_path: Path):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names: list[str] = []
        self.input_size: tuple[int, int] = (224, 224)
        self.model_sha256 = ""
        self._infer_fn = None  # compiled prediction function (about 9x faster than calling the model directly)
        self._grad_model = None  # set up only if Grad-CAM works for this model
        self._output_layer = None
        self._gradcam_fn = None
        # One prediction at a time: the model is not meant to be called from many threads at once
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def explain_available(self) -> bool:
        return self._grad_model is not None

    @property
    def model_version(self) -> str:
        return self.model_sha256[:12]

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load class names and the Keras model. Called once at startup."""
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        if not self.class_names_path.is_file():
            raise ModelLoadError(f"Class names file not found: {self.class_names_path}")

        try:
            with open(self.class_names_path) as file:
                class_names = [str(name).upper() for name in json.load(file)]
        except (OSError, ValueError) as error:
            raise ModelLoadError(f"Could not read class names: {error}") from error

        try:
            import keras  # imported here so a TensorFlow problem becomes a clean error

            # compile=False: we only run inference, and the saved loss is a custom function.
            model = keras.models.load_model(self.model_path, compile=False)
        except Exception as error:
            raise ModelLoadError(f"Could not load model: {error}") from error

        # Read the real input/output shapes from the model instead of hardcoding them
        _, height, width, channels = model.input_shape
        num_outputs = model.output_shape[-1]
        if channels != 3:
            raise ModelLoadError(f"Expected a 3-channel input, model has {channels}.")
        if num_outputs != len(class_names):
            raise ModelLoadError(
                f"Model has {num_outputs} outputs but class_names has {len(class_names)} entries."
            )

        self.model = model
        self.class_names = class_names
        self.input_size = (height, width)
        self.model_sha256 = file_sha256(self.model_path)
        self._compile_inference()
        self._prepare_gradcam()
        self._warm_up()
        logger.info(
            "Model loaded. version=%s input=%s classes=%s explain=%s",
            self.model_version, self.input_size, self.class_names, self.explain_available,
        )

    def _input_spec(self):
        import tensorflow as tf

        return tf.TensorSpec((1, *self.input_size, 3), tf.float32)  # we always send one image

    def _compile_inference(self) -> None:
        import tensorflow as tf

        def infer(batch):
            return self.model(batch, training=False)

        self._infer_fn = tf.function(infer, input_signature=[self._input_spec()], autograph=False)

    def _warm_up(self) -> None:
        """Run one dummy prediction so the first real user does not wait for TensorFlow to warm up."""
        self._infer_fn(np.zeros((1, *self.input_size, 3), dtype=np.float32))

    def _prepare_gradcam(self) -> None:
        """Build the helper model used for Grad-CAM. If anything looks unexpected, disable it."""
        try:
            import keras

            # Last layer that still outputs a feature map (batch, h, w, channels)
            conv_layer = next(layer for layer in reversed(self.model.layers) if len(layer.output.shape) == 4)
            output_layer = self.model.layers[-1]
            if not isinstance(output_layer, keras.layers.Dense):
                raise ValueError("last layer is not Dense")

            # Output the feature map AND the input of the final softmax layer. The gradient is
            # taken from the raw scores (logits): softmax saturates and would give blank heatmaps.
            self._grad_model = keras.Model(self.model.input, [conv_layer.output, output_layer.input])
            self._output_layer = output_layer

            # Safety check: our logits path must reproduce the model's real probabilities
            rng = np.random.default_rng(0)
            sample = rng.uniform(0, 255, (1, *self.input_size, 3)).astype(np.float32)
            _, features = self._grad_model(sample, training=False)
            logits = np.asarray(features) @ np.asarray(output_layer.kernel) + np.asarray(output_layer.bias)
            rebuilt = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
            actual = np.asarray(self._infer_fn(sample))
            if not np.allclose(rebuilt, actual, atol=1e-3):
                raise ValueError("rebuilt probabilities do not match the model output")

            # ...and a full heatmap must actually come out (this also warms Grad-CAM up)
            import tensorflow as tf

            self._gradcam_fn = tf.function(
                self._gradcam_graph,
                input_signature=[self._input_spec(), tf.TensorSpec((), tf.int32)],
                autograph=False,
            )
            heatmap = self._gradcam_heatmap(sample, class_index=0)
            if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
                raise ValueError("Grad-CAM produced an invalid heatmap")
        except Exception as error:
            logger.warning("Grad-CAM disabled: %s", error)
            self._grad_model = None
            self._output_layer = None
            self._gradcam_fn = None

    # ------------------------------------------------------------------ inference
    def _gradcam_graph(self, tensor, class_index):
        """Grad-CAM math (runs compiled): which regions pushed the model toward `class_index`."""
        import tensorflow as tf

        with tf.GradientTape() as tape:
            feature_map, features = self._grad_model(tensor, training=False)
            logits = tf.matmul(features, self._output_layer.kernel) + self._output_layer.bias
            class_score = logits[:, class_index]
        gradients = tape.gradient(class_score, feature_map)
        channel_weights = tf.reduce_mean(gradients, axis=(1, 2))  # one weight per channel
        heatmap = tf.reduce_sum(feature_map * channel_weights[:, None, None, :], axis=-1)[0]
        heatmap = tf.nn.relu(heatmap)  # keep only evidence FOR the class
        return tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))  # scale to 0-1

    def _gradcam_heatmap(self, tensor: np.ndarray, class_index: int) -> np.ndarray:
        """Return a 2-D heatmap (values 0-1) for the given class."""
        import tensorflow as tf

        return self._gradcam_fn(tensor, tf.constant(class_index, tf.int32)).numpy()

    def predict(self, image_bytes: bytes, explain: bool = False) -> dict:
        """Preprocess the image, run the model, and format the result."""
        if not self.is_ready:
            raise ModelLoadError("Model is not loaded.")

        prepared = prepare_image(image_bytes, self.input_size)  # raises InvalidImageError
        warnings = list(prepared.quality_warnings)
        explanation = None

        with self._lock:
            probabilities = np.asarray(self._infer_fn(prepared.tensor))[0]
            best_index = int(np.argmax(probabilities))
            if explain:
                if self.explain_available:
                    heatmap = self._gradcam_heatmap(prepared.tensor, best_index)
                    explanation = heatmap_overlay_base64(prepared.image, heatmap)
                    if heatmap.max() <= 0:
                        warnings.append("The heatmap is empty: no image region clearly supported this result.")
                else:
                    warnings.append("A heatmap could not be produced for this model.")

        reasons = assess_uncertainty(probabilities)
        return {
            "success": True,
            "predicted_class": self.class_names[best_index],
            "confidence": round(float(probabilities[best_index]), 6),
            "predictions": {
                name: round(float(prob), 6)
                for name, prob in zip(self.class_names, probabilities)
            },
            "is_uncertain": bool(reasons),
            "uncertainty_reasons": reasons,
            "warnings": warnings,
            "explanation_png_base64": explanation,
            "model_version": self.model_version,
            "disclaimer": DISCLAIMER,
        }
