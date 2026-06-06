from contextlib import asynccontextmanager
import os
import sys
from typing import Dict, Any, AsyncGenerator

sys.path.append(os.getcwd())

import uvicorn

from src.api.routers import init_routers
from src.core.classification import Classifier
from src.core.preprocessing import Preprocessor
from src.settings import custom_logger, SettingsManager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


logger = custom_logger("API")


# Context manager, inicializa el preprocesador y el clasificador cuando se levanta la app
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Function for loading the classifier, preprocessor and settings on startup

    Args:
        app: FastAPI application instance
    """
    try:
        logger.info("Starting up application...")
        settings = SettingsManager()
        classifier = Classifier(
            model_path=settings.MODEL_PATH,
            labels_path=settings.LABELS_PATH,
            batch_size=settings.BATCH_SIZE,
        )
        image_size = getattr(settings, "IMAGE_SIZE", None) or classifier.get_input_image_size()
        preprocessor = Preprocessor(image_size=tuple(image_size))

        app.state.settings = settings
        app.state.preprocessor = preprocessor
        app.state.classifier = classifier

        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Shutting down application...")



app = FastAPI(
    title="practico-4-2026",
    description="API para el practico 4 de la materia de ML en Producción",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para evitar problemas de seguridad se habilita cualquier origen de solicitud
    allow_methods=["*"],  # se debería restringir esto a los dominios necesarios
    allow_headers=["*"],
)
# Cargar routers, estos son /health y /classification/images
init_routers(app)


#Punto de entrada para correr la app con uvicorn, se puede correr con `uvicorn src.api.app:app --reload`
if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
