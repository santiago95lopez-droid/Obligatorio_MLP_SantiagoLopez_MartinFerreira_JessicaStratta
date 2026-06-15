import os
import zipfile
from io import BytesIO

from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from src.settings import custom_logger
from src.structs.payload import ImageResponsePayload


logger = custom_logger("Classification Router")

classification_router = APIRouter()


@classification_router.post("/images")
async def classify_images(
    request: Request, image: UploadFile = File(...)
) -> ImageResponsePayload:
    """
    Endpoint for classifying a single uploaded image

    Args:
        request: Request object
        image: UploadFile containing the image file

    Returns:
        ImageResponsePayload object containing the classified image
    """

    preprocessed_image = await request.app.state.preprocessor.preprocess_image(image)
    response = request.app.state.classifier.predict(preprocessed_image)
    return response


@classification_router.post("/predict-batch")
async def predict_batch(request: Request, archive: UploadFile = File(...)) -> dict:
    """Procesa un ZIP con imágenes y devuelve la predicción para cada archivo válido."""
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se requiere un archivo ZIP con extensión .zip")

    await archive.seek(0)
    archive_bytes = await archive.read()
    results = []

    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive_file:
            for member in archive_file.infolist():
                if member.is_dir():
                    continue

                filename = os.path.basename(member.filename)
                extension = os.path.splitext(filename)[1].lower()

                if not filename or extension not in allowed_extensions:
                    continue

                try:
                    with archive_file.open(member) as member_file:
                        file_bytes = member_file.read()

                    batch_image = UploadFile(filename=filename, file=BytesIO(file_bytes))
                    payload = await request.app.state.preprocessor.preprocess_image(batch_image)
                    response = request.app.state.classifier.predict(payload)
                    prediction = response.images[0].label

                    results.append({"filename": filename, "prediction": prediction})
                except Exception as exc:
                    results.append(
                        {
                            "filename": filename,
                            "error": f"Error procesando imagen: {str(exc)}",
                        }
                    )
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400, detail="Archivo ZIP inválido o corrupto"
        )

    return {"predictions": results}
