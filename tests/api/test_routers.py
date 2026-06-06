
import io
from PIL import Image


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classification_image_endpoint(client):
    image = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"image": ("test.png", buffer, "image/png")}
    response = client.post("/classification/images", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "modelohongos/modelo_hongos_mobilenet.keras"
    assert len(data["images"]) == 1
    assert data["images"][0]["filename"] == "test.png"
    assert data["images"][0]["label"] == "dummy"
    assert data["images"][0]["score"] == 0.99
    assert data["images"][0]["metadata"]["scores"]["dummy"] == 0.99