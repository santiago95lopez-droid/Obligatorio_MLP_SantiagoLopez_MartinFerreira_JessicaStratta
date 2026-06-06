from fastapi import APIRouter
from typing import Dict
from src.settings import custom_logger


logger = custom_logger("API Health")

health_router = APIRouter()


@health_router.get("/health")
def get_health() -> Dict[str, str]:
    """
    Endpoint for checking the health of the API server

    Returns:
        Dictionary containing the status of the API server
    """

    return {"status": "ok"}
