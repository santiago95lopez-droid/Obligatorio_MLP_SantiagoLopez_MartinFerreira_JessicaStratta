from __future__ import annotations

"""Fast pre-filter to detect likely mushroom images before main inference.

The filter uses MobileNetV2 pretrained on ImageNet and keyword matching over
top predicted labels to decide whether an image should continue in the pipeline.
"""

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
    """Heuristic image filter for mushroom presence detection.

    This component is intentionally conservative and is used as an early gate
    before the main domain classifier executes.
    """

    def __init__(self, image_size: Tuple[int, int] | None = None) -> None:
        """Initializes the pre-filter and loads MobileNetV2 weights.

        Args:
            image_size: Optional target size used for model preprocessing.
        """
        self.image_size = image_size or DEFAULT_IMAGE_SIZE
        self.model = MobileNetV2(weights="imagenet")

    def _prepare_image(self, image_bytes: bytes) -> np.ndarray:
        """Converts raw bytes into a MobileNetV2-compatible tensor.

        Args:
            image_bytes: Raw binary content from an uploaded image.

        Returns:
            np.ndarray: Batched and normalized tensor for MobileNetV2.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        x = np.asarray(image, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    def is_mushroom(self, image_bytes: bytes) -> bool:
        """Evaluates whether an image likely contains a mushroom.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            bool: True when mushroom-related keywords appear in top predictions.
        """
        try:
            preprocessed_tensor = self._prepare_image(image_bytes)
            predictions = self.model.predict(preprocessed_tensor, verbose=0)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            # Match top-3 ImageNet labels against mushroom-related keywords.
            return any(
                any(keyword in label.lower() for keyword in MUSHROOM_KEYWORDS)
                for _, label, _ in decoded_predictions
            )
        except Exception as exc:
            logger.error("Error processing mushroom filter image: %s", exc)
            return False
