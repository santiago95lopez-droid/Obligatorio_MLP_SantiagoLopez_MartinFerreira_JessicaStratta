
import base64
import io
import zipfile
from io import BytesIO

import numpy as np
from PIL import Image

from src.structs.images import ClassifiedImage, ScoresMetadata
from src.structs.payload import ImageResponsePayload


class DummyPreprocessor:
    def __init__(self, image_size=None):
        self.image_size = image_size

    async def preprocess_image(self, image, filename=None):
        pixel_values = np.zeros((1, 224, 224, 3), dtype=np.float32)
        return {
            "filename": filename or getattr(image, "filename", "test.png"),
            "pixel_values": pixel_values,
        }


class DummyClassifier:
    def __init__(self, model_path, labels_path, batch_size):
        self.model_id = model_path
        self.labels_path = labels_path
        self.batch_size = batch_size

    def predict(self, payload):
        return ImageResponsePayload(
            images=[
                ClassifiedImage(
                    filename=payload["filename"],
                    label="dummy",
                    score=0.99,
                    metadata=ScoresMetadata(scores={"dummy": 0.99}),
                )
            ],
            model_id=self.model_id,
        )


class DummyMushroomFilter:
    def __init__(self, is_mushroom_result=True):
        self.is_mushroom_result = is_mushroom_result

    def is_mushroom(self, image_bytes):
        return self.is_mushroom_result


def _make_png_bytes(color="red", size=(10, 10)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def test_health_endpoint(client):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classification_image_endpoint(client):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter()

    files = {"image": ("test.png", io.BytesIO(_make_png_bytes("red")), "image/png")}
    response = client.post("/classification/images", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "dummy-model.tflite"
    assert len(data["images"]) == 1
    assert data["images"][0]["filename"] == "test.png"
    assert data["images"][0]["label"] == "dummy"
    assert data["images"][0]["score"] == 0.99
    assert data["images"][0]["metadata"]["scores"]["dummy"] == 0.99


def test_classification_rejects_non_mushroom_with_clear_message(client):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter(is_mushroom_result=False)

    files = {"image": ("test.png", io.BytesIO(_make_png_bytes("red")), "image/png")}
    response = client.post("/classification/images", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "No es un hongo"
    assert data["score"] == 0.0
    assert data["message"] == "No es un hongo"


def test_classification_image_endpoint_with_heatmap(client, monkeypatch):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter()

    class DummyExplainer:
        def __init__(self, model_path):
            self.model_path = model_path

        def explain_image(self, image_bytes):
            return b"fake-heatmap"

        def encode_heatmap(self, image_bytes):
            return base64.b64encode(self.explain_image(image_bytes)).decode("ascii")

    monkeypatch.setattr("src.api.routers.classification.GradCAMExplainer", DummyExplainer)

    files = {"image": ("test.png", io.BytesIO(_make_png_bytes("red")), "image/png")}
    response = client.post(
        "/classification/images",
        files=files,
        data={"generate_heatmap": "true"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["heatmap_base64"] == base64.b64encode(b"fake-heatmap").decode("ascii")


def test_predict_batch_endpoint_returns_predictions_for_each_archive_file(client):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter()

    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("first.png", _make_png_bytes("red"))
        archive.writestr("second.png", _make_png_bytes("blue"))

    archive_buffer.seek(0)
    files = {"archive": ("images.zip", archive_buffer, "application/zip")}
    response = client.post("/classification/predict-batch", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"] == [
        {"filename": "first.png", "prediction": "dummy"},
        {"filename": "second.png", "prediction": "dummy"},
    ]


def test_predict_batch_endpoint_rejects_invalid_archive(client):
    client.app.state.preprocessor = DummyPreprocessor(image_size=(224, 224))
    client.app.state.classifier = DummyClassifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    client.app.state.mushroom_filter = DummyMushroomFilter()

    files = {"archive": ("bad.zip", BytesIO(b"not-a-zip"), "application/zip")}
    response = client.post("/classification/predict-batch", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Archivo ZIP inválido o corrupto"