"""Streamlit frontend. It only talks to the FastAPI backend over HTTP - no model is loaded here."""
import base64
import os

import requests
import streamlit as st

# Backend address. Change it with:  SKIN_API_BASE_URL=http://host:port streamlit run streamlit_app.py
API_BASE_URL = os.getenv("SKIN_API_BASE_URL", "http://localhost:8000").rstrip("/")
# Only needed if the backend was started with SKIN_API_KEY set
API_KEY = os.getenv("SKIN_API_KEY", "")
PREDICT_URL = f"{API_BASE_URL}/api/v1/predict"
HEALTH_URL = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 60

# Friendly names, only used for display
CLASS_DESCRIPTIONS = {
    "NV": "Melanocytic nevus (common mole)",
    "MEL": "Melanoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis-like lesion",
}

DISCLAIMER = (
    "**Medical disclaimer:** This is an educational decision-support prototype. "
    "It is **not** a clinically validated diagnostic tool. The model's confidence is a "
    "statistical score, **not** a confirmed medical diagnosis. Always consult a qualified "
    "dermatologist about any skin concern."
)


def auth_headers() -> dict:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def check_backend_health() -> tuple[bool, str]:
    """Return (is_healthy, message) for the sidebar status."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        body = response.json()
    except requests.exceptions.RequestException:
        return False, "Backend is not reachable."
    except ValueError:
        return False, "Backend returned an unexpected response."
    if not body.get("model_loaded"):
        return False, "Backend is online but the model is not loaded."
    if body.get("auth_required") and not API_KEY:
        return False, "Backend is online but needs an API key. Start this app with SKIN_API_KEY set."
    return True, "Backend online, model loaded."


def request_prediction(
    image_bytes: bytes, filename: str, content_type: str, explain: bool = False
) -> tuple[dict | None, str | None]:
    """Send the image to the backend. Returns (result, error_message); exactly one is None."""
    try:
        response = requests.post(
            PREDICT_URL,
            files={"file": (filename, image_bytes, content_type)},
            params={"explain": "true"} if explain else None,
            headers=auth_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to the backend at {API_BASE_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond. Please try again."
    except requests.exceptions.RequestException as error:
        return None, f"Request failed: {error}"

    try:
        body = response.json()
    except ValueError:
        return None, f"Backend returned an unexpected response (HTTP {response.status_code})."

    if response.status_code != 200:
        detail = body.get("detail", "Unknown error")
        if isinstance(detail, list):  # FastAPI validation errors come as a list
            detail = "; ".join(item.get("msg", "invalid request") for item in detail)
        if response.status_code == 429 and response.headers.get("Retry-After"):
            detail += f" Try again in about {response.headers['Retry-After']} seconds."
        return None, f"{detail} (HTTP {response.status_code})"

    if not all(key in body for key in ("predicted_class", "confidence", "predictions")):
        return None, "Backend response is missing expected fields."
    return body, None


def show_result(result: dict) -> None:
    predicted = result["predicted_class"]
    uncertain = result.get("is_uncertain", False)

    if uncertain:
        reasons = "\n".join(f"- {reason}" for reason in result.get("uncertainty_reasons", []))
        st.warning(f"**Uncertain result.** The model is not sure about this image.\n\n{reasons}\n\n"
                   "Treat the output below with extra caution.")
    for warning in result.get("warnings", []):
        st.info(warning)

    st.subheader("Model output")
    col_class, col_conf = st.columns(2)
    col_class.metric("Most likely class" if uncertain else "Predicted class", predicted)
    col_conf.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    st.caption(CLASS_DESCRIPTIONS.get(predicted, ""))

    st.markdown("**Confidence for each class**")
    for name, probability in sorted(result["predictions"].items(), key=lambda item: item[1], reverse=True):
        label = f"{name} - {CLASS_DESCRIPTIONS.get(name, '')}"
        st.progress(min(max(probability, 0.0), 1.0), text=f"{label}: {probability * 100:.1f}%")

    if result.get("explanation_png_base64"):
        with st.expander("Where did the model look? (Grad-CAM heatmap)", expanded=True):
            st.image(base64.b64decode(result["explanation_png_base64"]), width="stretch")
            st.caption(
                "Warm colours mark the image regions that influenced this result most. This shows how the "
                "model behaved, it is not a medical finding. It can highlight irrelevant things such as "
                "rulers, hair or image borders."
            )

    st.warning(result.get("disclaimer") or DISCLAIMER)
    if result.get("model_version"):
        st.caption(f"Model version: {result['model_version']}")


def main() -> None:
    st.set_page_config(page_title="Skin Disease Detection System", page_icon="🔬", layout="centered")

    with st.sidebar:
        st.header("Backend status")
        healthy, message = check_backend_health()
        (st.success if healthy else st.error)(message)
        st.caption(f"API: {API_BASE_URL}")

    st.title("Skin Disease Detection System")
    st.write(
        "Upload a dermoscopic skin image and a trained deep-learning model will estimate which of "
        "four lesion categories it most resembles: **NV, MEL, BCC, BKL**."
    )
    st.info(DISCLAIMER)

    uploaded = st.file_uploader("Choose a dermoscopic image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.session_state.pop("outcome", None)  # forget the old result when the image is removed
        return

    # A different image was chosen: forget the previous result
    file_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:
        st.session_state["file_id"] = file_id
        st.session_state.pop("outcome", None)

    image_bytes = uploaded.getvalue()
    try:
        st.image(image_bytes, caption=uploaded.name, width="stretch")
    except Exception:
        # A renamed or corrupted file can pass the extension check but cannot be drawn.
        # Keep going: the backend validates the file and will explain the problem.
        st.warning("This file could not be previewed - it may not be a valid image. "
                   "You can still press Predict and the server will check it.")

    want_heatmap = st.checkbox("Also show where the model looked (heatmap)", value=True)
    if st.button("Predict", type="primary"):
        with st.spinner("Analysing image..."):
            # Saved in session_state so the result stays on screen when the page reruns
            st.session_state["outcome"] = request_prediction(
                image_bytes, uploaded.name, uploaded.type or "image/jpeg", explain=want_heatmap
            )

    outcome = st.session_state.get("outcome")
    if outcome:
        result, error = outcome
        if error:
            st.error(error)
        else:
            show_result(result)


if __name__ == "__main__":
    main()
