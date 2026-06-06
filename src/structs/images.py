from typing import Dict

from pydantic import BaseModel


class ScoresMetadata(BaseModel):
    """Metadata for the scores of the image classification labels."""

    scores: Dict[str, float]


class ClassifiedImage(BaseModel):
    """Result for a classified image."""

    filename: str
    label: str
    score: float
    metadata: ScoresMetadata
