from typing import Dict, Any

import numpy as np

from src.settings import custom_logger
from src.structs.images import ClassifiedImage, ScoresMetadata
from src.structs.payload import ImageResponsePayload
from src.utils.file_loading import read_json_file


class Classifier:
    """Class for handling image classification using a local Keras model."""

    def __init__(self, model_path: str, labels_path: str, batch_size: int):
        self.logger = custom_logger(self.__class__.__name__)
        self.model_path = model_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.model_id = model_path

        self.logger.info(f"Loading Keras model from {self.model_path}")
        self.load_model()

    def load_model(self) -> None:
        """Load a Keras image classification model and its labels."""
        self.model = self._load_keras_model(self.model_path)
        self.labels = self._load_labels(self.labels_path)
        self.num_labels = len(self.labels)
        self.logger.info(
            f"Keras model loaded successfully with {self.num_labels} labels"
        )

    def _load_keras_model(self, model_path: str):
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import mobilenet
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load the Keras model. "
                "Install it with `pip install tensorflow`."
            ) from exc

        custom_objects = {
            "preprocess_input": mobilenet.preprocess_input,
        }

        return tf.keras.models.load_model(
            model_path, compile=False, custom_objects=custom_objects
        )

    def _load_labels(self, labels_path: str) -> list[str]:
        labels_data = read_json_file(labels_path)

        if isinstance(labels_data, dict):
            try:
                sorted_items = sorted(labels_data.items(), key=lambda item: int(item[0]))
            except ValueError:
                sorted_items = sorted(labels_data.items())
            return [label for _, label in sorted_items]

        if isinstance(labels_data, list):
            return labels_data

        raise TypeError(
            f"Labels file {labels_path} must contain a list or dict of labels"
        )

    def get_input_image_size(self) -> tuple[int, int]:
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if not input_shape or len(input_shape) != 4:
            raise RuntimeError(
                "Unsupported Keras model input shape for image classification."
            )

        if input_shape[-1] == 3:
            return int(input_shape[1]), int(input_shape[2])
        if input_shape[1] == 3:
            return int(input_shape[2]), int(input_shape[3])

        raise RuntimeError(
            "Unsupported Keras model input shape. Expected channels-last or channels-first."
        )

    def predict(self, payload: Dict[str, Any]) -> ImageResponsePayload:
        pixel_values = payload["pixel_values"]
        pixel_values = np.asarray(pixel_values, dtype=np.float32)

        predictions = self.model.predict(
            pixel_values, batch_size=self.batch_size, verbose=0
        )
        scores = np.asarray(predictions)

        if scores.ndim == 1:
            scores = scores[np.newaxis, ...]

        if scores.shape[-1] != self.num_labels:
            raise RuntimeError(
                f"Model output size {scores.shape[-1]} does not match {self.num_labels} labels."
            )

        if not np.allclose(np.sum(scores, axis=-1), 1.0, atol=1e-3):
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        top_idx = int(np.argmax(scores[0]))
        top_label = self.labels[top_idx]

        image_result = ClassifiedImage(
            filename=payload["filename"],
            label=top_label,
            score=float(scores[0][top_idx]),
            metadata=ScoresMetadata(
                scores={
                    self.labels[j]: float(scores[0][j])
                    for j in range(self.num_labels)
                }
            ),
        )

        return ImageResponsePayload(images=[image_result], model_id=self.model_id)
