import base64
import os
from io import BytesIO
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.settings import custom_logger


class GradCAMExplainer:
    """Generate Grad-CAM heatmaps for a MobileNetV2-based image classifier."""

    def __init__(self, model_path: Optional[str] = None, image_size: tuple[int, int] = (224, 224)) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.image_size = image_size
        self.model_path = model_path or os.getenv(
            "KERAS_MODEL_PATH",
            "modelohongos/modelo_hongos_mobilenet.keras",
        )
        self.model = self._load_model(self.model_path)

        self.base_model = self.model
        for layer in self.model.layers:
            if layer.__class__.__name__ in ["Functional", "Model", "Sequential"]:
                self.base_model = layer
                break

        self.last_conv_layer = self._find_last_conv_layer(self.base_model)

    def _load_model(self, model_path: str):
        try:
            from tensorflow.keras.applications import mobilenet_v2
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to run Grad-CAM explanations.") from exc

        self.logger.info(f"Loading Keras model for Grad-CAM from {model_path}")
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"preprocess_input": mobilenet_v2.preprocess_input},
        )

    def _find_last_conv_layer(self, model) -> tf.keras.layers.Layer:
        preferred_names = ["out_relu", "Conv_1"]
        for layer_name in preferred_names:
            try:
                layer = model.get_layer(layer_name)
                if layer is not None:
                    return layer
            except ValueError:
                continue

        for layer in reversed(model.layers):
            layer_class_name = layer.__class__.__name__
            if "Conv2D" in layer_class_name or "Conv" in layer_class_name:
                return layer

            output_shape = getattr(layer, "output_shape", None)
            if isinstance(output_shape, (list, tuple)) and len(output_shape) == 4:
                return layer

        raise RuntimeError("No convolutional layer was found in the loaded model.")

    def _prepare_image(self, image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize(self.image_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(pil_image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return image_array, pil_image

    def _make_heatmap(self, image_array: np.ndarray, class_idx: int) -> np.ndarray:
        try:
            grad_model = tf.keras.Model(
                inputs=self.base_model.input,
                outputs=[self.last_conv_layer.output, self.base_model.output],
            )

            img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, preds = grad_model(img_tensor, training=False)
                class_channel = preds[:, class_idx]

            grads = tape.gradient(class_channel, conv_outputs)

            if grads is None:
                raise ValueError("Gradients are still None after extracting base_model.")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
            heatmap = tf.maximum(heatmap, 0.0)
            heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
            return heatmap.numpy()
        except Exception as e:
            self.logger.error(f"Grad-CAM error: {str(e)}")
            return np.zeros(image_array.shape[1:3], dtype=np.float32)

    def _overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray) -> bytes:
        heatmap_image = np.uint8(255 * heatmap)
        heatmap_image = Image.fromarray(heatmap_image).resize(original_image.size, Image.Resampling.LANCZOS)
        heatmap_array = np.array(heatmap_image)

        try:
            import cv2
        except ImportError:
            cv2 = None

        if cv2 is not None:
            colored_heatmap = cv2.applyColorMap(heatmap_array, cv2.COLORMAP_JET)
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            original_array = np.array(original_image)
            blended = 0.4 * colored_heatmap + 0.6 * original_array
            blended = np.clip(blended, 0, 255).astype(np.uint32)
            overlay = Image.fromarray(blended.astype(np.uint8))
        else:
            heatmap_pil = Image.fromarray(heatmap_array).convert("L")
            overlay = Image.blend(original_image.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=0.4)

        output = BytesIO()
        overlay.save(output, format="PNG")
        return output.getvalue()

    def explain_image(self, image_bytes: bytes, target_class_idx: Optional[int] = None) -> bytes:
        image_array, original_image = self._prepare_image(image_bytes)
        predictions = self.model.predict(image_array, verbose=0)[0]
        if target_class_idx is None:
            target_class_idx = int(np.argmax(predictions))

        heatmap = self._make_heatmap(image_array, target_class_idx)
        return self._overlay_heatmap(original_image, heatmap)

    def encode_heatmap(self, image_bytes: bytes) -> str:
        heatmap_bytes = self.explain_image(image_bytes)
        return base64.b64encode(heatmap_bytes).decode("ascii")
