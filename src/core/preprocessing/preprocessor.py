"""Image preprocessing utilities used before model inference.

This module normalizes heterogeneous image inputs (UploadFile, bytes, and
file-like objects) into a standardized tensor payload.
"""

from io import BytesIO
from typing import Any, Sequence

import numpy as np
from fastapi import UploadFile
from PIL import Image

from src.settings import custom_logger


class Preprocessor:
    """Preprocesses image payloads into model-ready tensors."""

    def __init__(self, image_size: Sequence[int] | None = None) -> None:
        """Initializes preprocessing configuration.

        Args:
            image_size: Optional target size as (width, height).
        """
        self.logger = custom_logger(self.__class__.__name__)
        self.image_size = tuple(image_size) if image_size is not None else (224, 224)

    async def preprocess_image(
        self,
        image: UploadFile | bytes | bytearray | BytesIO | Any,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Preprocesses a single image into a normalized batched tensor.

        Args:
            image: UploadFile, bytes, bytearray, BytesIO, or another file-like object.
            filename: Optional filename override to preserve in the payload.

        Returns:
            dict[str, Any]: Dictionary with filename and `pixel_values` tensor.

        Raises:
            TypeError: If the input type is unsupported.
        """
        if isinstance(image, UploadFile):
            self.logger.info(f"Preprocessing image {image.filename}")
            await image.seek(0)
            image_bytes = await image.read()
            filename = filename or image.filename
        elif isinstance(image, (bytes, bytearray)):
            self.logger.info("Preprocessing image bytes payload")
            image_bytes = bytes(image)
        elif isinstance(image, BytesIO):
            self.logger.info(
                f"Preprocessing image file-like object {getattr(image, 'name', 'unknown')}"
            )
            image.seek(0)
            image_bytes = image.read()
        elif hasattr(image, "read"):
            self.logger.info(
                f"Preprocessing image file-like object {getattr(image, 'name', 'unknown')}"
            )
            if hasattr(image, "seek"):
                image.seek(0)
            image_bytes = image.read()
            # Support async-compatible file-like implementations without changing control flow.
            if hasattr(image_bytes, "__await__"):
                image_bytes = await image_bytes
        else:
            raise TypeError(
                "Unsupported image payload type. Expected UploadFile, bytes, bytearray, or a file-like object."
            )

        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self.image_size:
            pil_image = pil_image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Keep normalization contract in [0, 1] and always enforce a batch dimension.
        image_array = np.asarray(pil_image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        return {
            "filename": filename or "",
            "pixel_values": image_array,
        }
