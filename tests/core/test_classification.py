import numpy as np
from src.core.classification import Classifier


class DummyInterpreter:
    def __init__(self):
        self.input_details = [{"index": 0, "shape": [1, 224, 224, 3]}]
        self.output_details = [{"index": 1}]
        self.input_tensor = None

    def get_input_details(self):
        return self.input_details

    def get_output_details(self):
        return self.output_details

    def set_tensor(self, index, tensor):
        self.input_tensor = tensor

    def invoke(self):
        return None

    def get_tensor(self, index):
        return np.array([[0.1, 0.9]], dtype=np.float32)


def test_classifier_predict(monkeypatch):
    monkeypatch.setattr(
        "src.core.classification.classifier.Classifier._load_tflite_model",
        lambda self, path: DummyInterpreter(),
    )
    monkeypatch.setattr(
        "src.core.classification.classifier.read_json_file",
        lambda path: {"0": "cat", "1": "dog"},
    )

    classifier = Classifier(
        model_path="dummy-model.tflite",
        labels_path="dummy-labels.json",
        batch_size=1,
    )
    payload = {
        "filename": "test.png",
        "pixel_values": np.zeros((1, 224, 224, 3), dtype=np.float32),
    }
    result = classifier.predict(payload)

    assert result.model_id == "dummy-model.tflite"
    assert len(result.images) == 1
    assert result.images[0].filename == "test.png"
    assert result.images[0].label == "dog"
    assert abs(result.images[0].score - 0.9) < 1e-6
    assert abs(result.images[0].metadata.scores["cat"] - 0.1) < 1e-6
    assert abs(result.images[0].metadata.scores["dog"] - 0.9) < 1e-6
