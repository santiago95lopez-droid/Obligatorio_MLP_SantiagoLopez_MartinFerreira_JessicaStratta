from typing import List, Optional

from pydantic import BaseModel

from src.structs.images import ClassifiedImage


class ImageResponsePayload(BaseModel):
    """Response payload for the image classification endpoint."""

    images: List[ClassifiedImage]
    model_id: str
    heatmap: Optional[str] = None
