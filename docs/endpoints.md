# Endpoints Documentation

## 1. Example of Endpoint for Image Classification

### Description

This endpoint allows users to classify a single image and receive predictions with associated labels, confidence scores, and metadata.

### Endpoint

```
POST /classification/images
```

### Request

Upload a single image file in multipart/form-data format:

```bash
curl -X POST http://localhost:8080/classification/images \
  -F "image=@test.png"
```

### Response Body

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

### Notes

- The request must include a single image file under the `image` field.
- The response returns predictions for the uploaded image, including the top label, score, and raw label probabilities.
- The `model_id` field identifies the model used to generate the prediction.
