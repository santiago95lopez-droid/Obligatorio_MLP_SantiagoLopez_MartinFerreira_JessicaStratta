import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
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
    def is_mushroom(self, image_bytes):
        return True


@pytest.fixture(autouse=True)
def patch_app_classes(monkeypatch):
    monkeypatch.setattr("src.api.app.preprocessor", DummyPreprocessor(image_size=(224, 224)))
    monkeypatch.setattr(
        "src.api.app.classifier",
        DummyClassifier(model_path="dummy-model.tflite", labels_path="dummy-labels.json", batch_size=1),
    )
    monkeypatch.setattr("src.api.app.mushroom_filter", DummyMushroomFilter())


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


