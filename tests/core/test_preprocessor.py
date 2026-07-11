import asyncio
from io import BytesIO

from PIL import Image
from src.core.preprocessing.preprocessor import Preprocessor


def test_preprocessor():
    """Test preprocessing of byte and file-like image payloads."""

    preprocessor = Preprocessor(image_size=(224, 224))
    image = Image.new("RGB", (10, 10), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    bytes_result = asyncio.run(
        preprocessor.preprocess_image(image_bytes, filename="test.png")
    )
    assert bytes_result["filename"] == "test.png"
    assert "pixel_values" in bytes_result
    assert bytes_result["pixel_values"].shape == (1, 224, 224, 3)

    bytesio_result = asyncio.run(
        preprocessor.preprocess_image(BytesIO(image_bytes), filename="test.png")
    )
    assert bytesio_result["filename"] == "test.png"
    assert "pixel_values" in bytesio_result
    assert bytesio_result["pixel_values"].shape == (1, 224, 224, 3)
