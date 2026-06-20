from io import BytesIO
from typing import Sequence

import numpy as np
from fastapi import UploadFile
from PIL import Image

from src.settings import custom_logger


class Preprocessor:
    """Class for handling image preprocessing for local Keras models."""

    def __init__(self, image_size: Sequence[int] | None = None) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.image_size = tuple(image_size) if image_size is not None else (224, 224)

    async def preprocess_image(
        self,
        image: UploadFile | bytes | BytesIO,
        filename: str | None = None,
    ):
        """
        Method for preprocessing a single uploaded image.

        Args:
            image: UploadFile, bytes, or file-like object containing the image
            filename: Optional filename to preserve in the payload

        Returns:
            A dictionary containing the filename and prepared pixel values
        """
        if isinstance(image, UploadFile):
            self.logger.info(f"Preprocessing image {image.filename}")
            image_bytes = await image.read()
            filename = filename or image.filename
        elif isinstance(image, (bytes, bytearray)):
            self.logger.info("Preprocessing image bytes payload")
            image_bytes = bytes(image)
        else:
            self.logger.info(
                f"Preprocessing image file-like object {getattr(image, 'name', 'unknown')}"
            )
            image_bytes = image.read()

        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self.image_size:
            pil_image = pil_image.resize(self.image_size, Image.Resampling.LANCZOS)

        image_array = np.asarray(pil_image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return {
            "filename": filename or "",
            "pixel_values": image_array,
        }
