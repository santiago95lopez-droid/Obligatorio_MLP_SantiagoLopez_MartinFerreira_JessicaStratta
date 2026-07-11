# Project Context and Session State

## 1. Current Status
As of 2026-07-11, the FastAPI stack for the mushroom classifier is fully integrated with a TensorFlow Lite quantized model and verified end to end through pytest for both single-image classification and batch prediction flows.

## 2. Purpose & Context
This repository implements a mushroom classification service for uploaded images. The system is designed to:
- accept one or more uploaded images through FastAPI endpoints,
- reject obvious non-mushroom content with a MobileNetV2-based mushroom filter,
- preprocess images to a fixed RGB shape suitable for inference,
- run inference with a local classifier and return a predicted label plus confidence scores.

The technical approach is a small production-style ML serving stack:
- FastAPI for the HTTP API.
- Pydantic-style response models for structured predictions.
- A preprocessing layer that resizes and normalizes images.
- A classifier wrapper that loads and runs a local model artifact.
- A settings manager and logging layer for deployment and debugging.

Scope note: this repository is primarily an inference-serving project and an assignment-style ML deployment example; it is not a full training pipeline with a dedicated training notebook or dataset folder in the current snapshot.

## 3. Project Structure
Key locations in this repository:

- root/
  - README.md: high-level project description and usage instructions.
  - IMPLEMENTATION_DETAILS.md: notes on the TensorFlow Lite migration approach.
  - MIGRATION_TFLITE_SUMMARY.md: migration summary and rollback notes.
  - CONTEXT.md: this file, the technical memory and session state snapshot.
  - requirements.txt, pyproject.toml, dev-requirements.txt: Python dependencies.
  - Dockerfile, docker-compose.yml, run_api.bat: deployment helpers.

- src/
  - api/app.py: FastAPI application startup and lifespan configuration.
  - api/routers/classification.py: image classification and batch prediction routes.
  - api/routers/health.py: health endpoint.
  - core/classification/classifier.py: classifier wrapper for model loading and inference.
  - core/classification/mushroom_filter.py: image-based prefilter using MobileNetV2.
  - core/preprocessing/preprocessor.py: image resizing and normalization.
  - settings/settings_manager.py: configuration loading from YAML.
  - settings/settings.yml: runtime model path, class labels, and inference settings.
  - structs/: request/response payload definitions.
  - utils/file_loading.py: JSON/YAML loading helpers.

- modelohongos/
  - clases_hongos.json: class label mapping.
  - modelo_hongos_mobilenet.keras: legacy Keras model artifact.
  - modelo_quantizado.tflite: current TensorFlow Lite model artifact.

- models/
  - local cache and model-related assets used by the filtering and serving stack.

- tests/
  - api/test_routers.py: router and endpoint integration checks.
  - core/test_classification.py: classifier behavior tests.
  - core/test_preprocessor.py: preprocessing behavior tests.

- docs/
  - endpoint documentation and example payloads.

Note: no explicit training script, notebook, or dataset directory is currently present in this repository snapshot.

## 4. What's Working (Facts)
The following items are currently supported by the repository and were verified or directly observed from the current implementation:
- The FastAPI app boots successfully and the health endpoint returns a 200 response.
- The app exposes image classification and batch prediction routes through the router layer.
- The classifier implementation has been updated to use a TensorFlow Lite interpreter workflow via tf.lite.Interpreter.
- The preprocessing pipeline resizes images to a fixed size and normalizes them to floating-point values in the $[0,1]$ range.
- The label mapping is driven by the JSON file under modelohongos/ and is used to translate model outputs into human-readable classes.
- The stale Keras-style test references were removed, and the classifier-related tests were aligned with the TFLite interface and input/output shapes.
- The preprocessing path now handles UploadFile, bytes, and BytesIO/file-like payloads deterministically by resetting streams and passing bytes to the downstream preprocessor layer.
- Both the single-image classification endpoint and the batch prediction endpoint have been fully verified end to end using pytest.

## 5. What Was Attempted & Failed (Lessons Learned)
These are the important migration and integration lessons learned during the project:
- The migration from the original Keras model artifact to a quantized TensorFlow Lite model required updating the classifier tests and fixtures to match the new interpreter-based interface.
- Stream handling for uploaded files was a recurring source of issues, and the final fix was to reset the upload stream and pass normalized byte payloads into the downstream preprocessing and inference layers.

## 6. Next Steps
Priority order for the next development session:
1. Benchmark the quantized TensorFlow Lite model for inference latency and memory usage on the target environment.
2. Prepare deployment artifacts or environment-specific run instructions based on the verified FastAPI setup.
3. Optionally expand documentation with the verified endpoint behavior and performance observations.

## 7. Instructions for Future AIs
Treat this file as the authoritative state snapshot for the project. Before making changes, read this file first, understand the current status, and preserve the intent of the existing architecture. When a change affects the model pipeline, API contract, or tests, update this file immediately so the next session does not lose context. Prefer verification with pytest or a local run over assumption, and treat old references to the Keras-only loading path as migration leftovers until they are explicitly revalidated.
