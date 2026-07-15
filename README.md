# Servido de Modelos - Clasificador de Hongos con TFLite y Grad-CAM

Este proyecto implementa una API REST para clasificación de imágenes de hongos con FastAPI. La inferencia principal se ejecuta localmente con un modelo cuantizado TensorFlow Lite, y el artefacto `.keras` se conserva únicamente para respaldar la generación opcional de mapas de atención Grad-CAM. El modelo clasifica en tres categorías: `Comestible`, `No comestible` y `Venenoso`.

## Resumen

- API FastAPI para clasificación individual y por lotes.
- Inferencia local eficiente en CPU mediante `tf.lite.Interpreter` sobre `modelo_quantizado.tflite`.
- Filtro previo de hongos basado en MobileNetV2 para cortar temprano imágenes no válidas.
- Explicabilidad opcional con Grad-CAM usando `tf.GradientTape` y el artefacto legado `.keras`.
- Interfaz Streamlit para probar imágenes individuales o archivos ZIP.

## Estructura del proyecto

```text
.
├── app_streamlit.py
├── docker-compose.yml
├── Dockerfile
├── docs/
│   ├── endpoints.md
│   └── examples.json
├── modelohongos/
│   ├── clases_hongos.json
│   ├── modelo_hongos_mobilenet.keras
│   └── modelo_quantizado.tflite
├── run_api.bat
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── classification.py
│   │       └── health.py
│   ├── core/
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py
│   │   │   ├── explainability.py
│   │   │   └── mushroom_filter.py
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       └── preprocessor.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── settings.yml
│   │   └── settings_manager.py
│   ├── structs/
│   │   ├── images.py
│   │   └── payload.py
│   └── utils/
│       └── file_loading.py
├── tests/
│   ├── api/
│   │   └── test_routers.py
│   └── core/
│       ├── test_classification.py
│       └── test_preprocessor.py
├── requirements.txt
└── pyproject.toml
```

## Tecnologías principales

### FastAPI

FastAPI expone los endpoints, valida requests multipart y permite documentar automáticamente la API en Swagger/OpenAPI.

### TensorFlow Lite como motor principal

La ejecución de inferencia productiva se realiza con `tf.lite.Interpreter` cargando `modelohongos/modelo_quantizado.tflite`. Esta es la ruta principal del sistema porque prioriza:

- menor costo de inferencia en CPU,
- artefactos cuantizados más livianos,
- despliegue local simple sin depender de un servidor externo de modelos.

### Keras solo para explicabilidad

El archivo `modelohongos/modelo_hongos_mobilenet.keras` no es el motor principal de predicción. Se mantiene estrictamente para el módulo opcional de explicabilidad, donde `tf.GradientTape` calcula Grad-CAM sobre el backbone MobileNetV2 y devuelve un heatmap codificado en base64.

## Pipeline de imágenes

La API sigue un pipeline robusto de tres etapas:

### 1. Mushroom pre-filter

Antes de preprocesar o clasificar, `src/core/classification/mushroom_filter.py` analiza los bytes crudos de la imagen con MobileNetV2. Si la imagen no parece contener un hongo, la API corta el flujo de forma temprana y responde `200 OK` con:

```json
{
   "label": "No es un hongo",
   "score": 0.0,
   "message": "No es un hongo"
}
```

### 2. Preprocessing

Si la imagen supera el filtro, `src/core/preprocessing/preprocessor.py`:

- convierte la imagen a RGB,
- la redimensiona a `(224, 224)`,
- normaliza los píxeles al rango `[0, 1]`,
- agrega dimensión de batch para el intérprete TFLite.

### 3. Quantized inference y explainability

`src/core/classification/classifier.py` ejecuta inferencia local con TFLite y genera:

- la clase top-1 (`Comestible`, `No comestible` o `Venenoso`),
- su score,
- el diccionario completo de probabilidades por etiqueta,
- el `model_id` asociado al artefacto TFLite usado.

Si el cliente envía `generate_heatmap=true`, el router además instancia `src/core/classification/explainability.py`, construye un submodelo funcional sobre el `.keras`, extrae gradientes con `tf.GradientTape` y devuelve un mapa de atención Grad-CAM codificado  dentro de la respuesta.

## Endpoints

### `GET /health`

Chequeo simple de salud de la API.

### `POST /classification/images`

Clasifica una imagen individual recibida como `multipart/form-data` en el campo `image`.

Parámetros:

- `image`: archivo `.png`, `.jpg` o `.jpeg`.
- `generate_heatmap`: boolean opcional. Si vale `true`, agrega el mapa Grad-CAM a la respuesta.

Ejemplo:

```bash
curl -X POST "http://localhost:8080/classification/images?generate_heatmap=true" \
   -F "image=@test.png"
```

### `POST /classification/predict-batch`

Procesa un archivo `.zip` con múltiples imágenes válidas y devuelve una predicción por archivo, incluyendo confianza (`score`).

Ejemplo:

```bash
curl -X POST http://localhost:8080/classification/predict-batch \
   -F "archive=@imagenes.zip"
```

## Ejemplo de respuesta JSON

Respuesta realista para una clasificación individual con `generate_heatmap=true`:

```json
{
   "images": [
      {
         "filename": "hongo_1.png",
         "label": "Comestible",
         "score": 0.9473,
         "metadata": {
            "scores": {
               "Comestible": 0.9473,
               "No comestible": 0.0312,
               "Venenoso": 0.0215
            }
         }
      }
   ],
   "model_id": "modelohongos/modelo_quantizado.tflite",
   "heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

Respuesta realista para una imagen descartada por el pre-filtro:

```json
{
   "label": "No es un hongo",
   "score": 0.0,
   "message": "No es un hongo"
}
```

Respuesta realista para batch:

```json
{
   "predictions": [
      {
         "filename": "muestra_1.png",
         "prediction": "Comestible",
         "score": 0.9473
      },
      {
         "filename": "muestra_2.png",
         "prediction": "No es un hongo!",
         "score": 0.0
      },
      {
         "filename": "muestra_3.png",
         "prediction": "Error",
         "score": 0.0,
         "error": "Error procesando imagen: cannot identify image file"
      }
   ]
}
```

## Testing

La solución queda cubierta por suites `pytest` que validan el comportamiento esperado de extremo a extremo y por componente:

- clasificación de imagen individual,
- predicción batch vía ZIP,
- short-circuit del filtro de hongos,
- preprocesamiento y forma del tensor normalizado,
- contrato de respuesta del router.

Ejecución de tests:

```bash
pytest
```

## Ejecución local

### Opción 1: entorno virtual con `pip`

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```

La documentación interactiva queda disponible en `http://127.0.0.1:8080/docs`.

### Opción 2: usando Poetry

```bash
poetry install
poetry run python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
```

### Opción 3: helper `run_api.bat`

El script `run_api.bat` automatiza el flujo en Windows:

- verifica Poetry,
- crea el entorno local dentro del proyecto,
- instala dependencias,
- valida la existencia de `modelo_hongos_mobilenet.keras` y `modelo_quantizado.tflite`,
- levanta la API en `:8080`,
- abre además la interfaz Streamlit.

Ejecutar:

```bat
run_api.bat
```

## Docker

### Docker directo

```bash
docker build -t mushroom-classifier-api .
docker run -p 8080:8080 mushroom-classifier-api
```

### Docker Compose

```bash
docker-compose up --build
```

Para detener los servicios:

```bash
docker-compose down
```

## Notas finales

- La configuración por defecto apunta a `modelohongos/modelo_quantizado.tflite` y `modelohongos/clases_hongos.json` desde `src/settings/settings.yml`.
- El tamaño de entrada esperado por el pipeline es `(224, 224)`.
- La app FastAPI y la UI Streamlit permiten probar tanto clasificación simple como batch desde el mismo repositorio.



- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Docker Documentation](https://docs.docker.com/)
