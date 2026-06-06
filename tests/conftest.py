
import pytest
try:
    import torch
except Exception:
    torch = None
    import numpy as np
from fastapi.testclient import TestClient
from src.api.app import app
from src.structs.images import ClassifiedImage, ScoresMetadata
from src.structs.payload import ImageResponsePayload


class DummyPreprocessor:
    def __init__(self, image_size=None):
        self.image_size = image_size

    async def preprocess_image(self, image):
        if torch is not None:
            pixel_values = torch.zeros((1, 224, 224, 3))
        else:
            pixel_values = np.zeros((1, 224, 224, 3))
        return {
            "filename": image.filename,
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


@pytest.fixture(autouse=True)
def patch_app_classes(monkeypatch):
    monkeypatch.setattr("src.api.app.Preprocessor", DummyPreprocessor)
    monkeypatch.setattr("src.api.app.Classifier", DummyClassifier)


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


