from __future__ import annotations

import logging
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

logger = logging.getLogger(__name__)

MUSHROOM_KEYWORDS = [
    "mushroom",
    "fungus",
    "agaric",
    "boletus",
    "puffball",
    "chanterelle",
    "morel",
    "truffle",
    "coral fungus",
    "earthstar",
    "stinkhorn",
    "toadstool",
    "bolete",
]
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)


class MushroomFilter:
    """Filter for identifying whether an image likely contains a mushroom."""

    def __init__(self, image_size: Tuple[int, int] | None = None) -> None:
        self.image_size = image_size or DEFAULT_IMAGE_SIZE
        self.model = MobileNetV2(weights="imagenet")

    def _prepare_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        x = np.asarray(image, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    def is_mushroom(self, image_bytes: bytes) -> bool:
        try:
            x = self._prepare_image(image_bytes)
            preds = self.model.predict(x, verbose=0)
            decoded = decode_predictions(preds, top=3)[0]

            return any(
                any(keyword in label.lower() for keyword in MUSHROOM_KEYWORDS)
                for _, label, _ in decoded
            )
        except Exception as exc:
            logger.error("Error processing mushroom filter image: %s", exc)
            return False
