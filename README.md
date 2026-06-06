[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hKg-8jSY)
# Práctico 4: Model Serving

En este práctico se implementará un sistema de clasificación de imágenes utilizando un modelo Keras/TensorFlow y FastAPI para crear una API REST.

## Contenido

- Usamos un sistema de clasificación de imágenes usando un modelo Keras/TensorFlow
- Creamos una API REST con FastAPI para servir predicciones
- Realizamos pruebas de inferencia y calidad del modelo


## Estructura del Proyecto

```
practico-4-2026/
├── src/                     # Código fuente
│   ├── api/                # Endpoints de la API
│   │   ├── routers/       # Definición de rutas
│   │   └── app.py         # Aplicación FastAPI
│   ├── core/              # Lógica principal
│   │   ├── classification.py
│   │   └── preprocessing.py
│   ├── settings/          # Configuración
│   └── utils/             # Utilidades
├── tests/                 # Tests unitarios
├── Dockerfile            # Configuración de Docker (Extra)
├── docker-compose.yml    # Configuración de Docker Compose (Extra)
└── requirements.txt      # Dependencias del proyecto
```

## 1. FastAPI y TensorFlow/Keras

### FastAPI

FastAPI es un framework moderno y de alto rendimiento para construir APIs con Python. Sus principales ventajas son:

- **Rendimiento**: Uno de los frameworks más rápidos disponibles, comparable a NodeJS y Go
- **Documentación Automática**: Genera automáticamente documentación interactiva (Swagger/OpenAPI)
- **Validación de Tipos**: Integración nativa con el sistema de tipos de Python
- **Async/Await**: Soporte nativo para operaciones asíncronas
- **Fácil de Usar**: Sintaxis intuitiva y minimalista
- **Basado en Estándares**: Compatible con OpenAPI y JSON Schema

### TensorFlow / Keras

El proyecto usa TensorFlow/Keras para cargar y servir un modelo local (.keras). Ventajas:

- **Modelo Ligero**: Fácil de exportar y desplegar
- **Compatibilidad**: Integración nativa con `tensorflow` y `keras`
- **GPU/CPU**: Soporte para aceleración cuando está disponible

## 2. Configuración del Entorno

### Configuración del Entorno de Desarrollo

1. Crear entorno virtual:

   ```bash
   # macOS/Linux
   virtualenv venv
   source venv/bin/activate

   # Windows
   virtualenv venv
   .\venv\Scripts\activate
   ```
2. Instalar dependencias:

   ```bash
   # Instalar dependencias principales
   pip install -r requirements.txt

   # Instalar dependencias de desarrollo
   pip install -r dev-requirements.txt
   ```

Para levantar el servidor de forma local, usar el comando de Poetry:


```bash
poetry run python src/api/app.py

```
Si no usa Poetry, el comando es:

```bash
python src/api/app.py

```
También se puede crear un contenedor de Docker para ejecutar el práctico, más información al final de este documento.

## 3. Clasificación de Imágenes

La clasificación de imágenes consiste en asignar una etiqueta a una imagen cargada en la API. En este proyecto, utilizamos modelos de Hugging Face para procesar la imagen y devolver la probabilidad de cada clase.

### Procesamiento de Imágenes

El pipeline de procesamiento incluye:

1. **Carga de Imagen**: Leer el archivo cargado por el usuario
2. **Conversión a RGB**: Normalizar los canales de color
3. **Preprocesamiento**: Redimensionar y convertir a tensores compatibles con el modelo
4. **Clasificación**: Ejecutar inferencia y generar scores

### Batch Processing

Para optimizar el rendimiento, se puede procesar múltiples imágenes en lotes:

- Procesar varias imágenes en una sola pasada
- Reducir overhead de inferencia
- Mejorar throughput del sistema


## 4. Ejemplo de Resultados

El sistema es capaz de clasificar imágenes y devolver etiquetas y scores. Por ejemplo, al enviar una imagen con el siguiente comando:

```bash
curl -X POST http://localhost:8080/classification/images \
  -F "image=@test.png"
```

El sistema responde con la clasificación detallada:

```json
{
  "images": [
    {
      "filename": "test.png",
      "label": "dog",
      "score": 0.92,
      "metadata": {
        "scores": {
          "cat": 0.08,
          "dog": 0.92
        }
      }
    }
  ],
   "model_id": "modelohongos/modelo_hongos_mobilenet.keras"
}
```

La respuesta incluye:

- El nombre del archivo cargado
- La etiqueta predicha (`label`)
- El score de confianza para la predicción
- Metadatos con los scores para todas las clases
- El ID del modelo utilizado


## 5. Testing

El proyecto incluye tests de integración para asegurar la calidad del código:

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

```

### Tipos de Tests

- **API**: Endpoints, respuestas y manejo de errores
- **Clasificación**: Precisión del modelo y casos límite
- **Preprocesamiento**: Limpieza y normalización de texto

### Mejores Prácticas

- Usar fixtures de pytest para configuración
- Mockear dependencias externas
- Probar casos límite y errores

## 6. Ejercicio Práctico

### Clasificación de Imágenes

Implementa un systema de clasificación de imágenes, usando TensorFlow/Keras. Recomendaciones:

- Buscar modelos que no tengan un peso muy grande (<800M de parámetros).
- Verificar el tiempo de inferencia.
- Priorizar modelos que funcionen bien en CPU.

#### Tareas:

1. Crear nuevo endpoint para la clasificación de imágenes.
2. Probar con diferentes inputs de imágenes.

#### Entregables:

1. Código adaptado
2. README con:
   - Modelo usado y su tamaño
   - Resultados de ejemplos de clasificación de imágenes.
   - Tiempos de inferencia
  
Opcionalmente puede agregar preprocesamiento a las imágenes de las requests antes de hacer la clasificación.

## 7. Recursos Adicionales




### Configuración de Docker

Hay dos formas de ejecutar la aplicación con Docker:

##### Opción 1: Docker Directo

1. Construir la imagen:

   ```bash
   docker build -t text-classification-api .
   ```
2. Ejecutar el contenedor:

   ```bash
   # Ejecutar en modo detached
   docker run -d -p 8080:8080 text-classification-api

   # Ver logs
   docker logs -f <container_id>
   ```

##### Opción 2: Docker Compose

Docker Compose es una herramienta para definir y ejecutar aplicaciones multi-contenedor. Sus principales ventajas son:

- **Definición de Servicios**: Permite definir todos los servicios necesarios en un archivo YAML
- **Entorno Aislado**: Cada servicio corre en su propio contenedor
- **Desarrollo Consistente**: Garantiza que todos los desarrolladores usen la misma configuración
- **Fácil Despliegue**: Un solo comando para levantar toda la aplicación
- **Gestión de Dependencias**: Maneja automáticamente las dependencias entre servicios
- **Variables de Entorno**: Centraliza la configuración de variables de entorno
- **Volúmenes**: Facilita el manejo de datos persistentes y desarrollo en tiempo real

1. Ejecutar con Docker Compose (construye la imagen automáticamente):

   ```bash
   # Ejecutar en modo detached
   docker-compose up -d

   # Ver logs
   docker-compose logs -f
   ```
2. Para detener los contenedores:

   ```bash
   # Si usaste Docker directo
   docker stop <container_id>

   # Si usaste Docker Compose
   docker-compose down
   ```



- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Docker Documentation](https://docs.docker.com/)
