from fastapi import APIRouter, Request, UploadFile, File

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
