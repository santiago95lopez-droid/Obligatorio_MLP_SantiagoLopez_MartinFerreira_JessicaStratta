# Endpoints Documentation

## GET /health

### Description

Health check endpoint.

### Response

```json
{
  "status": "ok"
}
```

## POST /classification/images

### Description

Classifies one image and optionally returns a Grad-CAM heatmap.

### Request

Multipart form-data fields:
- `image` (required): image file (`.png`, `.jpg`, `.jpeg`)
- `generate_heatmap` (optional query param): `true` or `false`

```bash
curl -X POST "http://localhost:8080/classification/images?generate_heatmap=true" \
  -F "image=@test.png"
```

### Success Response

```json
{
  "images": [
    {
      "filename": "test.png",
      "label": "Comestible",
      "score": 0.92,
      "metadata": {
        "scores": {
          "Comestible": 0.92,
          "No comestible": 0.05,
          "Venenoso": 0.03
        }
      }
    }
  ],
  "model_id": "modelohongos/modelo_quantizado.tflite",
  "heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Non-Mushroom Response

```json
{
  "label": "No es un hongo",
  "score": 0.0,
  "message": "No es un hongo"
}
```

## POST /classification/predict-batch

### Description

Processes a ZIP archive and returns one result per valid image member.

### Request

Multipart form-data fields:
- `archive` (required): `.zip` file

```bash
curl -X POST "http://localhost:8080/classification/predict-batch" \
  -F "archive=@imagenes.zip"
```

### Response

```json
{
  "predictions": [
    {
      "filename": "muestra_1.png",
      "prediction": "Comestible",
      "score": 0.94
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

### Notes

- Only `.jpg`, `.jpeg`, and `.png` files inside the ZIP are processed.
- Invalid/corrupted ZIP uploads return HTTP 400 with detail.
- Batch output keeps a consistent per-item shape by always including `prediction` and `score`.
