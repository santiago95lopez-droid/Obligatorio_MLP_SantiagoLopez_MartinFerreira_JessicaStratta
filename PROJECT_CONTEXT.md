# Proyecto: scrapper_fungiatlas

## Ubicación
`C:\Users\marti\Desktop\Master\H3\Machine Learning en Produccion\scrapper_fungiatlas`

## Propósito general
Este proyecto está diseñado para extraer imágenes de hongos de Fungiatlas (`fungiatlas.com`) y clasificar las imágenes descargadas en carpetas de "ok" y "ruido" utilizando un clasificador basado en el modelo preentrenado MobileNetV2 de ImageNet.

## Dependencias
El proyecto usa Python 3.10+ con dependencias declaradas en `pyproject.toml`:
- beautifulsoup4
- requests
- tensorflow
- pillow

## Archivos principales

### `scrapper2.py`
- Versión refactorizada del scraper.
- Usa como punto de entrada `https://fungiatlas.com/all-mushrooms/`.
- Extrae enlaces de especies desde el índice alfabético del sitio.
- Procesa cada especie y busca imágenes en componentes tipo carrusel/galería.
- Descarga imágenes a una única carpeta plana: `hongos_scrapp`.
- Usa nombres tipo `species_name_1.jpg`, `species_name_2.jpg` y sanitiza caracteres incompatibles con Windows.
- Usa filtros básicos para evitar descargar logos, avatares y SVG.

### `scaraper01.py`
- Versión inicial del scraper.
- Navega directamente cada sección y descarga imágenes encontradas en el contenido principal.
- Guarda las imágenes en la carpeta de sección correspondiente.
- Usa un naming sencillo `hongo_<contador>.jpg`.

### `clasificador_honho_nohongo.py`
- Clasifica las imágenes descargadas usando `MobileNetV2(weights='imagenet')`.
- Define palabras clave de hongos (`MUSHROOM_KEYWORDS`) para identificar si una imagen es de hongo.
- Procesa únicamente las imágenes que se encuentran directamente en la carpeta `hongos_scrapp`.
- Mantiene las imágenes clasificadas como hongos en la raíz de `hongos_scrapp`.
- Mueve las imágenes clasificadas como ruido a la subcarpeta `hongos_scrapp/ruido`.

### `audit.py`
- Realiza una auditoría manual automática de hasta 10 imágenes seleccionadas al azar desde las carpetas `ruido`.
- Para cada imagen, muestra las 3 predicciones superiores de ImageNet.
- Permite revisar si las imágenes clasificadas como ruido contienen realmente objetos no asociados a hongos.

## Estructura de carpetas
- `hongos_scrapp/`
  - imágenes descargadas en formato plano
  - `ruido/` para las imágenes clasificadas como no hongos
- `hongos_scrapeados.zip` (archivo comprimido existente con hongos ya recolectados)

## Configuración de proyecto
### `pyproject.toml`
- Nombre del proyecto: `scrapper`
- Descripción: `scrappear la pagina fungiatlas`
- Autor: `martinferreirab`
- Requiere Python >= 3.10
- Usa Poetry como sistema de empaquetado/build.

### `.gitignore`
- Ignora imágenes dentro de las carpetas de hongos.
- Ignora `__pycache__/` y `.venv/`.

## Estado actual
- El flujo principal ya fue adaptado a una arquitectura plana con `hongos_scrapp`.
- El scraper descarga imágenes desde el índice alfabético de Fungiatlas hacia `hongos_scrapp`.
- El clasificador procesa las imágenes de esa carpeta y mueve el ruido a `hongos_scrapp/ruido`.
- No hay pruebas automatizadas registradas.
- El flujo de trabajo actual es:
  1. Ejecutar `scrapper2.py` para descargar imágenes.
  2. Ejecutar `clasificador_honho_nohongo.py` para separar imágenes válidas y ruido.
  3. Ejecutar `audit.py` para revisar manualmente las imágenes enviadas a `ruido`.

## Notas importantes
- El clasificador no está entrenado para hongos específicamente; usa un modelo de ImageNet y palabras clave heurísticas.
- Las carpetas `ok` y `ruido` pueden contener resultados erróneos por la naturaleza del enfoque actual.
- El scraper puede necesitar ajustes si Fungiatlas cambia su estructura HTML.

## Cómo usar este documento
Este archivo está pensado para dar contexto a otra IA o colaborador. Contiene:
- objetivos del proyecto
- scripts disponibles
- estructura de carpetas
- estado actual

> Mantener este documento actualizado con cada cambio realizado en el repositorio.

## Historial de actualizaciones
- 2026-07-10: Creado el documento con estado inicial del proyecto.
- 2026-07-10: Se refactorizó el scraper para usar `https://fungiatlas.com/all-mushrooms/` y guardar imágenes en `hongos_scrapp` con nombres basados en la especie.
- 2026-07-10: Se adaptó el clasificador para procesar únicamente las imágenes de `hongos_scrapp` y mover el ruido a `hongos_scrapp/ruido`.
- 2026-07-10: El flujo actual quedó alineado con la arquitectura plana nueva del proyecto.
