# Project Issues and How We Solved Them

This document summarizes the main issues encountered during the project and the actions taken to resolve them.

## 1. Migration from Keras to TFLite without breaking the API

Issue:
- The project moved from a .keras model to a quantized .tflite model, but clients (FastAPI consumers and Streamlit UI) still needed the same high-level behavior.

Root cause:
- Inference internals changed from model.predict() to TFLite interpreter calls.

How we solved it:
- Replaced Keras loading with tf.lite.Interpreter.
- Added interpreter.allocate_tensors() during initialization.
- Implemented TFLite inference flow: set_tensor() -> invoke() -> get_tensor().
- Preserved output assembly through existing payload structures.

Result:
- Faster and lighter inference path while keeping endpoint behavior stable for consumers.

## 2. TFLite initialization and tensor metadata access errors

Issue:
- Typical runtime errors included missing tensor metadata or invalid tensor usage.

Root cause:
- Not allocating tensors before accessing input_details/output_details or invoking the model.

How we solved it:
- Enforced allocate_tensors() right after interpreter creation.
- Stored input_details and output_details once at model load time.

Result:
- Stable access to tensor shape/index metadata and consistent inference execution.

## 3. Input tensor shape mismatch

Issue:
- Some payloads could fail when input dimensionality did not match model expectations.

Root cause:
- Variability in preprocessed payloads and differences between 3D and 4D tensor assumptions.

How we solved it:
- Normalized input to float32.
- Ensured batch dimension is present (1, H, W, 3) before inference.
- Kept preprocessor contract at image size (224, 224) with RGB conversion.

Result:
- Consistent inference input format and fewer shape-related runtime failures.

## 4. Non-normalized output scores (logits vs probabilities)

Issue:
- Model output could be logits instead of probabilities.

Root cause:
- Quantized models can expose raw scores depending on export/conversion settings.

How we solved it:
- Added a probability check (sum close to 1.0).
- Applied manual softmax when needed.

Result:
- Reliable and interpretable confidence values in responses.

## 5. Streamlit batch confidence showing N/A

Issue:
- In batch predictions, the Streamlit table displayed N/A for confidence.

Root cause:
- Batch endpoint returned prediction label but not score for successful items.

How we solved it:
- Updated POST /classification/predict-batch to always include score.
- Extracted score from response.images[0].score.
- Cast score to native Python float to avoid JSON serialization issues with NumPy types.

Result:
- Streamlit now shows confidence percentages correctly for batch results.

## 6. Inconsistent batch response schema across scenarios

Issue:
- Success, non-mushroom, and error items had different structures, complicating frontend handling.

Root cause:
- Error and non-mushroom branches omitted fields expected by the UI.

How we solved it:
- Standardized per-item batch payload shape.
- Non-mushroom items now include prediction and score (0.0).
- Error items now include prediction: Error, score: 0.0, and an error detail.

Result:
- Uniform contract for batch consumers and simpler rendering logic.

## 7. Documentation drift from actual behavior

Issue:
- Some markdown docs showed outdated examples (species labels, old model_id, old endpoint examples, missing batch score).

Root cause:
- Documentation was not fully synchronized after migration and endpoint refinements.

How we solved it:
- Updated core documentation files to reflect current contracts and examples:
  - README.md
  - CONTEXT.md
  - docs/endpoints.md
  - QUICK_START_GUIDE.md
  - IMPLEMENTATION_DETAILS.md
  - MIGRATION_TFLITE_SUMMARY.md
- Clarified class meaning: Comestible, No comestible, Venenoso.
- Updated batch examples to include score and error consistency.

Result:
- Documentation now matches implementation and tests, reducing onboarding and reporting confusion.

## 8. Explainability model role confusion

Issue:
- Risk of treating the .keras model as the primary serving model.

Root cause:
- Coexistence of two model artifacts in the same repository.

How we solved it:
- Explicitly documented and preserved architecture:
  - .tflite is the production inference model.
  - .keras is used only for optional Grad-CAM generation.

Result:
- Clear separation of concerns between serving and explainability paths.

## 9. Filter ordering in the pipeline

Issue:
- Potential regressions if filtering is applied after preprocessing/inference.

Root cause:
- Pipeline sequencing is easy to change accidentally during refactors.

How we solved it:
- Kept mushroom filter check on raw image bytes before preprocessing and classification.
- Documented this as a critical architectural rule in project context.

Result:
- Early rejection remains efficient and behaviorally consistent.

## Lessons Learned

- Keep runtime contracts explicit and consistent across all branches, not only happy paths.
- Always serialize API numbers as native Python types when NumPy is involved.
- For model migrations, keep interface stability and isolate internal changes.
- Update tests and docs in the same change cycle to prevent contract drift.
- Document architectural intent (serving model vs explainability model) to avoid future regressions.
