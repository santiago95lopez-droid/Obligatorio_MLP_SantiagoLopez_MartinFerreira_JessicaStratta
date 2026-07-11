# Project Context and Session State

## 1. Current Status
As of 2026-07-11, this repository is a finalized model-serving project for mushroom image classification built with FastAPI, TensorFlow Lite, and an optional Grad-CAM explainability path. The production inference route is based on a quantized `.tflite` artifact, while the legacy `.keras` artifact is retained only for explanation generation.

## 2. Purpose and Scope
This project serves mushroom image predictions through HTTP endpoints and a lightweight Streamlit UI. The system is designed to:
- accept single uploaded images through FastAPI,
- accept ZIP archives for batch prediction,
- reject non-mushroom images early using a MobileNetV2-based pre-filter,
- preprocess valid images to the expected tensor format,
- run local quantized inference on CPU via TensorFlow Lite,
- optionally return a Grad-CAM attention heatmap for single-image requests.

This is an inference-serving repository, not a training repository. There is no dedicated training pipeline, dataset directory, or notebook-based experiment flow in the current snapshot.

## 3. Core Architecture

### Main inference path
- `src/core/classification/classifier.py` is the main classifier wrapper.
- It loads `modelohongos/modelo_quantizado.tflite` through `tf.lite.Interpreter`.
- It loads label names from `modelohongos/clases_hongos.json`.
- It returns a structured payload with top-1 label, confidence score, per-class scores, and `model_id`.

### Optional explainability path
- `src/core/classification/explainability.py` contains `GradCAMExplainer`.
- It loads `modelohongos/modelo_hongos_mobilenet.keras` only when a heatmap is requested.
- It uses `tf.GradientTape` and a functional sub-model to compute Grad-CAM.
- It returns a base64-encoded PNG heatmap in the response field `heatmap`.

### Pre-filter path
- `src/core/classification/mushroom_filter.py` runs before preprocessing and inference.
- It receives raw image bytes.
- It uses MobileNetV2 ImageNet predictions and mushroom-related keywords to decide whether the image likely contains a mushroom.
- If the filter rejects the image, the single-image endpoint returns HTTP 200 with the explicit payload:

```json
{
  "label": "No es un hongo",
  "score": 0.0,
  "message": "No es un hongo"
}
```

### Preprocessing path
- `src/core/preprocessing/preprocessor.py` converts images to RGB.
- It resizes them to `(224, 224)`.
- It normalizes pixel values to the `[0, 1]` range.
- It adds the batch dimension expected by the TFLite interpreter.

## 4. Project Structure Snapshot

Important locations:

- root/
  - `README.md`: updated project overview and execution instructions.
  - `CONTEXT.md`: authoritative implementation snapshot for future sessions.
  - `IMPLEMENTATION_DETAILS.md`: migration and implementation notes.
  - `MIGRATION_TFLITE_SUMMARY.md`: TFLite migration notes.
  - `Dockerfile`, `docker-compose.yml`, `run_api.bat`: runtime helpers.
  - `requirements.txt`, `dev-requirements.txt`, `pyproject.toml`: dependency configuration.
  - `app_streamlit.py`: lightweight UI for single-image and ZIP testing.

- `modelohongos/`
  - `clases_hongos.json`: label mapping.
  - `modelo_hongos_mobilenet.keras`: legacy Keras artifact used only for Grad-CAM.
  - `modelo_quantizado.tflite`: production inference model.

- `src/api/`
  - `app.py`: FastAPI app startup, lifespan wiring, router registration.
  - `routers/classification.py`: single-image and batch classification routes.
  - `routers/health.py`: health route.

- `src/core/classification/`
  - `classifier.py`: TFLite inference wrapper.
  - `explainability.py`: Grad-CAM generation.
  - `mushroom_filter.py`: MobileNetV2 pre-filter.

- `src/core/preprocessing/`
  - `preprocessor.py`: resizing, normalization, tensor preparation.

- `src/settings/`
  - `settings.yml`: runtime defaults.
  - `settings_manager.py`: YAML-backed settings loader.

- `src/structs/`
  - `images.py`, `payload.py`: response models.

- `tests/`
  - `api/test_routers.py`: endpoint contracts and router integration.
  - `core/test_classification.py`: classifier behavior under mocked TFLite interpreter.
  - `core/test_preprocessor.py`: preprocessing tensor-shape and input-type coverage.

## 5. Runtime Contracts

### `GET /health`
- Returns a simple healthy response.

### `POST /classification/images`
- Accepts one image in multipart form under `image`.
- Accepts optional `generate_heatmap=true`.
- If the pre-filter rejects the image, returns the non-mushroom payload shown above.
- Otherwise returns an `ImageResponsePayload` with:
  - `images`
  - `model_id`
  - optional `heatmap`

### `POST /classification/predict-batch`
- Accepts a `.zip` archive under `archive`.
- Iterates valid `.jpg`, `.jpeg`, and `.png` members.
- Returns a `predictions` list with per-file outcomes.
- Non-mushroom files are returned as `"No es un hongo!"` in the batch path.

## 6. Verified Facts
The following points are directly supported by the current code and tests:
- The production model path in `src/settings/settings.yml` points to `modelohongos/modelo_quantizado.tflite`.
- The labels path points to `modelohongos/clases_hongos.json`.
- The expected image size is `(224, 224)`.
- The preprocessor supports `UploadFile`, raw `bytes`, `bytearray`, `BytesIO`, and file-like objects.
- The classifier converts payloads to `float32`, ensures a batch dimension, runs TFLite inference, and softmaxes outputs if needed.
- The single-image route can append a base64 heatmap when `generate_heatmap` is requested.
- The Streamlit app can call both the single-image and batch endpoints and render the heatmap next to the uploaded image.
- Router tests cover health, single-image classification, non-mushroom rejection, heatmap behavior, batch success, and invalid ZIP handling.
- Core tests cover classifier prediction behavior and preprocessing output shape.

## 7. Lessons Learned
- The repository has already completed a migration from direct Keras-serving assumptions to a TFLite-first inference architecture.
- The most important implementation detail is that the mushroom filter must operate on raw image bytes before preprocessing.
- Stream reset behavior matters when combining file upload reads with downstream preprocessing and explainability.
- The `.keras` artifact should be treated as an auxiliary explainability dependency, not as the primary serving model.

## 8. Execution Notes
Supported execution paths currently documented in the repository:
- local virtualenv plus `uvicorn`,
- Poetry-based local execution,
- `run_api.bat` for Windows automation,
- Docker via `Dockerfile`,
- Docker Compose via `docker-compose.yml`.

`run_api.bat` additionally installs dependencies through Poetry, validates both model artifacts, launches the FastAPI app, launches Streamlit, and opens Swagger UI.

## 9. Recommended Next Steps
If future work continues from this state, the highest-value follow-ups are:
1. Measure inference latency and memory usage of the quantized TFLite path on the target CPU.
2. Align secondary docs such as `docs/endpoints.md` with the final router contract if they still reference older payloads.
3. Add dedicated runtime verification for the full Grad-CAM path against the real `.keras` artifact if environment setup allows it.

## 10. Instructions for Future AIs
Read this file before making architectural or API-contract changes. Treat it as the current authoritative state of the repository. If a future change affects model artifacts, request/response payloads, endpoint behavior, preprocessing, explainability, or test coverage, update this file in the same session so the next agent inherits the correct project state.
