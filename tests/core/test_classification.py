import numpy as np
from src.core.classification import Classifier


class DummyModel:
    def predict(self, pixel_values, batch_size=None, verbose=0):
        return np.array([[0.1, 0.9]])


def test_classifier_predict(monkeypatch):
    monkeypatch.setattr(
        "src.core.classification.classifier.Classifier._load_keras_model",
        lambda self, path: DummyModel(),
    )
    monkeypatch.setattr(
        "src.core.classification.classifier.read_json_file",
        lambda path: {"0": "cat", "1": "dog"},
    )

    classifier = Classifier(
        model_path="dummy-model.keras",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    payload = {"filename": "test.png", "pixel_values": np.zeros((1, 224, 224, 3), dtype=np.float32)}
    result = classifier.predict(payload)

    assert result.model_id == "dummy-model.keras"
    assert len(result.images) == 1
    assert result.images[0].filename == "test.png"
    assert result.images[0].label == "dog"
    assert abs(result.images[0].score - 0.9) < 1e-6
    assert result.images[0].metadata.scores == {"cat": 0.1, "dog": 0.9}
