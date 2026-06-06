import asyncio
from io import BytesIO

from PIL import Image
from starlette.datastructures import UploadFile
from src.core.preprocessing.preprocessor import Preprocessor


def test_preprocessor():
    """Test preprocessing of an uploaded image with the local Keras preprocessor."""

    preprocessor = Preprocessor(image_size=(224, 224))
    image = Image.new("RGB", (10, 10), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    upload_file = UploadFile(filename="test.png", file=buffer)
    result = asyncio.run(preprocessor.preprocess_image(upload_file))

    assert result["filename"] == "test.png"
    assert "pixel_values" in result
    # Expect channels-last shape for Keras models: (batch, height, width, channels)
    assert result["pixel_values"].shape == (1, 224, 224, 3)
