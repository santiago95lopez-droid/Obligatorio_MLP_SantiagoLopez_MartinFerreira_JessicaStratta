"""TFLite classifier service used by the API layer.

This module encapsulates model loading, label loading, input shape validation,
and inference post-processing for the quantized TensorFlow Lite artifact.
"""

from typing import Dict, Any

import numpy as np

from src.settings import custom_logger
from src.structs.images import ClassifiedImage, ScoresMetadata
from src.structs.payload import ImageResponsePayload
from src.utils.file_loading import read_json_file


class Classifier:
    """Image classifier backed by a quantized TensorFlow Lite model.

    The class is responsible for:
    - loading the TFLite interpreter,
    - loading class labels,
    - validating input/output tensor assumptions,
    - returning a normalized API payload.
    """

    def __init__(self, model_path: str, labels_path: str, batch_size: int):
        """Initializes classifier configuration and loads model resources.

        Args:
            model_path: Filesystem path to the .tflite model artifact.
            labels_path: Filesystem path to the labels JSON file.
            batch_size: Batch size configured for serving context.
        """
        self.logger = custom_logger(self.__class__.__name__)
        self.model_path = model_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.model_id = model_path

        self.logger.info(f"Loading TFLite model from {self.model_path}")
        self.load_model()

    def load_model(self) -> None:
        """Loads TFLite interpreter tensors and label metadata."""
        self.interpreter = self._load_tflite_model(self.model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.labels = self._load_labels(self.labels_path)
        self.num_labels = len(self.labels)
        self.logger.info(
            f"TFLite model loaded successfully with {self.num_labels} labels"
        )

    def _load_tflite_model(self, model_path: str) -> Any:
        """Creates and allocates a TensorFlow Lite interpreter.

        Args:
            model_path: Path to the TFLite model file.

        Returns:
            Any: Initialized tf.lite.Interpreter instance.

        Raises:
            RuntimeError: If TensorFlow is not available.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load the TFLite model. "
                "Install it with `pip install tensorflow`."
            ) from exc
        # Allocate tensors once at startup to avoid per-request initialization overhead.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def _load_labels(self, labels_path: str) -> list[str]:
        """Loads labels from JSON supporting dict and list formats.

        Args:
            labels_path: Path to labels JSON.

        Returns:
            list[str]: Ordered list of labels aligned to model output indices.

        Raises:
            TypeError: If the JSON structure is neither dict nor list.
        """
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
        """Returns expected input image spatial dimensions.

        Returns:
            tuple[int, int]: Expected (height, width) from model input tensor.

        Raises:
            RuntimeError: If model input shape/channels are incompatible.
        """
        input_shape = self.input_details[0]["shape"]
        
        if len(input_shape) != 4:
            raise RuntimeError(
                f"Unsupported TFLite model input shape: {input_shape}. "
                "Expected 4D shape (batch, height, width, channels)."
            )
        
        # Shape is typically (1, height, width, 3) for images
        _batch_size, height, width, channels = input_shape
        
        if channels != 3:
            raise RuntimeError(
                f"Unsupported number of channels: {channels}. Expected 3."
            )
        
        return int(height), int(width)

    def predict(self, payload: Dict[str, Any]) -> ImageResponsePayload:
        """Runs TFLite inference and formats a single-image response payload.

        Args:
            payload: Preprocessed payload containing image tensor and filename.

        Returns:
            ImageResponsePayload: Prediction payload with top label and per-class scores.

        Raises:
            RuntimeError: If model output size does not match configured labels.
        """
        pixel_values = payload["pixel_values"]
        pixel_values = np.asarray(pixel_values, dtype=np.float32)
        
        # Ensure pixel_values is 4D (batch, height, width, channels)
        if pixel_values.ndim == 3:
            pixel_values = np.expand_dims(pixel_values, axis=0)
        
        # Set the input tensor
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            pixel_values
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get the output tensor
        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )
        scores = np.asarray(output_data, dtype=np.float32)
        
        # Ensure scores is 2D (batch, num_classes)
        if scores.ndim == 1:
            scores = scores[np.newaxis, ...]
        
        # Verify output matches number of labels
        if scores.shape[-1] != self.num_labels:
            raise RuntimeError(
                f"Model output size {scores.shape[-1]} does not match "
                f"{self.num_labels} labels."
            )
        
        # Apply softmax manually when the model returns logits instead of probabilities.
        if not np.allclose(np.sum(scores, axis=-1), 1.0, atol=1e-3):
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Get top prediction
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
