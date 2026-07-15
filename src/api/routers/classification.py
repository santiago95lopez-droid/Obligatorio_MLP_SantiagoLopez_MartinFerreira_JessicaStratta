"""HTTP routers for image classification workflows.

This module exposes endpoints for:
1. Single-image classification, with optional Grad-CAM generation.
2. Batch classification from ZIP archives.

The implementation intentionally delegates inference, preprocessing, and filtering
to application state components initialized at startup.
"""

import os
import zipfile
from io import BytesIO
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.core.classification.explainability import GradCAMExplainer
from src.settings import custom_logger
from src.structs.payload import ImageResponsePayload

logger = custom_logger("Classification Router")

classification_router = APIRouter()


def _build_heatmap_payload(response: ImageResponsePayload, heatmap_base64: str | None) -> dict:
    """Builds the serialized image response and conditionally appends a heatmap.

    Args:
        response: Domain response object returned by the classifier.
        heatmap_base64: Optional base64-encoded PNG heatmap.

    Returns:
        dict: JSON-serializable payload preserving the original response structure.
    """
    payload = response.model_dump()
    if heatmap_base64 is not None:
        payload["heatmap"] = heatmap_base64
    return payload


@classification_router.post("/images")
async def classify_images(
    request: Request,
    image: UploadFile = File(...),
    generate_heatmap: bool = False,
) -> ImageResponsePayload | dict | str:
    """Classifies a single uploaded image.

    Args:
        request: FastAPI request with initialized app state services.
        image: Uploaded image file.
        generate_heatmap: When True, generates and appends a Grad-CAM heatmap.

    Returns:
        ImageResponsePayload | dict | str:
            - Standard classification payload when the image is accepted.
            - A rejection payload when the mushroom filter detects non-mushroom content.
            - A dictionary with heatmap when explainability is requested.
    """

    image_bytes = await image.read()
    # The pre-filter uses raw bytes to reject non-mushroom inputs before preprocessing/inference.
    if not request.app.state.mushroom_filter.is_mushroom(image_bytes):
        return {
            "label": "No es un hongo",
            "score": 0.0,
            "message": "No es un hongo",
        }

    await image.seek(0)
    preprocessed_payload = await request.app.state.preprocessor.preprocess_image(
        BytesIO(image_bytes), filename=image.filename
    )
    classification_response = request.app.state.classifier.predict(preprocessed_payload)

    if generate_heatmap:
        # Grad-CAM keeps using the legacy Keras artifact and is only computed on demand.
        explainer = GradCAMExplainer(model_path="modelohongos/modelo_hongos_mobilenet.keras")
        heatmap_base64 = explainer.encode_heatmap(image_bytes)
        return _build_heatmap_payload(classification_response, heatmap_base64)

    return classification_response


@classification_router.post("/predict-batch")
async def predict_batch(request: Request, archive: UploadFile = File(...)) -> dict:
    """Processes a ZIP archive and predicts each valid image entry.

    Args:
        request: FastAPI request with initialized app state services.
        archive: Uploaded ZIP file containing candidate images.

    Returns:
        dict: Payload in the format ``{"predictions": [...]}`` where each item
        contains filename, prediction, and score; failed items include an error detail.

    Raises:
        HTTPException: If the uploaded file is not a ZIP or is corrupted.
    """
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png"}

    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo ZIP con extensión .zip")

    await archive.seek(0)
    archive_bytes = await archive.read()
    results: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as zip_archive:
            for archive_member in zip_archive.infolist():
                if archive_member.is_dir():
                    continue

                filename = os.path.basename(archive_member.filename)
                extension = os.path.splitext(filename)[1].lower()

                if not filename or extension not in allowed_extensions:
                    continue

                try:
                    with zip_archive.open(archive_member) as member_file:
                        file_bytes = member_file.read()

                    # Keep filter evaluation on raw bytes to preserve the existing pipeline contract.
                    if not request.app.state.mushroom_filter.is_mushroom(file_bytes):
                        results.append(
                            {
                                "filename": filename,
                                "prediction": "No es un hongo!",
                                "score": 0.0,
                            }
                        )
                        continue

                    preprocessed_payload = await request.app.state.preprocessor.preprocess_image(
                        BytesIO(file_bytes), filename=filename
                    )
                    classification_response = request.app.state.classifier.predict(preprocessed_payload)
                    prediction = classification_response.images[0].label
                    # Cast explicitly to native float to avoid JSON serialization issues with NumPy scalars.
                    score = float(classification_response.images[0].score)

                    results.append(
                        {
                            "filename": filename,
                            "prediction": prediction,
                            "score": score,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "filename": filename,
                            "prediction": "Error",
                            "score": 0.0,
                            "error": f"Error procesando imagen: {str(exc)}",
                        }
                    )
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400, detail="Archivo ZIP inválido o corrupto"
        )

    return {"predictions": results}
