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

    async def preprocess_image(self, image: UploadFile):
        """
        Method for preprocessing a single uploaded image.

        Args:
            image: UploadFile containing the image file

        Returns:
            A dictionary containing the filename and prepared pixel values
        """
        self.logger.info(f"Preprocessing image {image.filename}")
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self.image_size:
            pil_image = pil_image.resize(self.image_size, Image.Resampling.LANCZOS)

        image_array = np.asarray(pil_image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return {
            "filename": image.filename,
            "pixel_values": image_array,
        }
