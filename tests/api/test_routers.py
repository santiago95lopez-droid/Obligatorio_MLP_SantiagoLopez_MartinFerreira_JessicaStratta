
import io
import zipfile
from io import BytesIO

from PIL import Image


def _make_png_bytes(color="red", size=(10, 10)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classification_image_endpoint(client):
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


def test_predict_batch_endpoint_returns_predictions_for_each_archive_file(client):
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
    files = {"archive": ("bad.zip", BytesIO(b"not-a-zip"), "application/zip")}
    response = client.post("/classification/predict-batch", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Archivo ZIP inválido o corrupto"