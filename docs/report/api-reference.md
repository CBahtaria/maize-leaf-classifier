# API Reference

**Base URL (local):** `http://localhost`
**Base URL (production):** `https://<your-domain>`

All endpoints return JSON. Errors follow the schema:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE"
}
```

---

## POST `/predict`

Single-image binary classification.

- **Description:** Accepts one image and returns a `Healthy`/`Diseased`
  classification with confidence and processing latency.
- **Request:** `multipart/form-data`
  - Field name: `file`
  - Accepted types: `image/jpeg`, `image/jpg`, `image/png`, `image/webp`
  - Maximum size: 10 MB
  - Maximum dimension: 10 000 px per side
- **Rate limit:** 20 requests per minute per source IP. Exceeding returns 429
  with `Retry-After: 60`.

### Response — 200 OK

```json
{
  "label": "Healthy",
  "confidence": 0.92,
  "processing_time_ms": 47.3,
  "model_version": "1.0.0"
}
```

- `label`: one of `"Healthy"`, `"Diseased"`.
- `confidence`: P(Diseased | image), in `[0.0, 1.0]`. When `label == "Healthy"`,
  this is `1 − P(Diseased)`.
- `processing_time_ms`: wall-clock inference time, server-side.
- `model_version`: semantic version string of the currently loaded model.

### Errors

| Status | `error_code` | When |
|---|---|---|
| 400 | `INVALID_IMAGE` | File is corrupt, empty, or PIL cannot decode it. |
| 413 | `FILE_TOO_LARGE` | Body exceeds 10 MB. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | `Content-Type` is not in the JPEG/PNG/WebP whitelist. |
| 429 | `RATE_LIMIT_EXCEEDED` | More than 20 requests in the last 60 seconds from your IP. |
| 503 | `MODEL_NOT_LOADED` | API is up but the model failed to load (deployment issue). |

### curl example

```bash
curl -X POST http://localhost/predict \
  -F "file=@./samples/healthy_leaf.jpg" \
  -H "Accept: application/json"
```

Expected output:

```json
{"label":"Healthy","confidence":0.04,"processing_time_ms":48.1,"model_version":"1.0.0"}
```

(Note: `confidence` here is `P(Diseased) = 0.04`, hence the `Healthy` label.)

---

## POST `/predict/batch`

Classify up to 10 images in a single request.

- **Description:** Same model and preprocessing as `/predict`, but accepts a
  list of files. Useful for offline-collected scan queues that re-sync on
  reconnection.
- **Request:** `multipart/form-data`
  - Field name: `files` (repeated, **plural**)
  - Maximum: 10 files per request
  - Each file: same constraints as `/predict` (≤ 10 MB, allowed MIME, ≤ 10 000 px/side)
- **Rate limit:** Shares the 20/min/IP limit with `/predict`.

### Response — 200 OK

```json
{
  "predictions": [
    {"label":"Healthy","confidence":0.04,"processing_time_ms":47.1,"model_version":"1.0.0"},
    {"label":"Diseased","confidence":0.91,"processing_time_ms":46.3,"model_version":"1.0.0"}
  ],
  "total_images": 2,
  "total_time_ms": 95.4
}
```

- `predictions`: list of `PredictionResponse` objects in the same order as the
  uploaded files.
- `total_images`: count of images successfully classified.
- `total_time_ms`: wall-clock time across all inferences in this batch.

### Errors

| Status | `error_code` | When |
|---|---|---|
| 400 | `INVALID_IMAGE` | At least one image is corrupt or empty. |
| 413 | `FILE_TOO_LARGE` | Any single file exceeds 10 MB. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Any file has a disallowed MIME. |
| 422 | `BATCH_TOO_LARGE` | More than 10 files in the request. |
| 429 | `RATE_LIMIT_EXCEEDED` | Rate limit exceeded. |
| 503 | `MODEL_NOT_LOADED` | Model failed to load. |

### curl example

```bash
curl -X POST http://localhost/predict/batch \
  -F "files=@./samples/leaf1.jpg" \
  -F "files=@./samples/leaf2.jpg" \
  -F "files=@./samples/leaf3.jpg" \
  -H "Accept: application/json"
```

---

## GET `/health`

Liveness and readiness probe.

- **Description:** Returns whether the service is responsive and whether the
  model is loaded. Used by Docker, the blue-green deploy script, and any
  external uptime monitor.
- **Rate limit:** None.

### Response — 200 OK

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 1834.5
}
```

- `status`: `"ok"` if the model is loaded and inference is working; `"degraded"`
  if the model is not loaded (the process is alive but cannot serve predictions).
- `model_loaded`: boolean.
- `model_version`: semantic version of the loaded model, or `"unknown"` when
  degraded.
- `uptime_seconds`: process uptime.

### curl example

```bash
curl http://localhost/health
```

Expected output:

```json
{"status":"ok","model_loaded":true,"model_version":"1.0.0","uptime_seconds":1834.5}
```

---

## GET `/model/info`

Returns metadata about the currently loaded model.

- **Description:** Static information for debugging and for the frontend to
  display under a Settings panel.
- **Rate limit:** None.

### Response — 200 OK

```json
{
  "architecture": "MobileNetV2",
  "version": "1.0.0",
  "tflite_size_mb": 3.5,
  "accuracy": 0.94,
  "sensitivity": 0.92,
  "specificity": 0.95,
  "auc_roc": 0.97,
  "model_path": "/app/model_artifacts/mobilenetv2_best.tflite"
}
```

- `architecture`: one of `MobileNetV2`, `Xception`, `InceptionV3`, `VGG16`,
  `ResNet50`.
- `version`: semantic version string.
- `tflite_size_mb`: size of the TFLite artifact on disk.
- `accuracy`, `sensitivity`, `specificity`, `auc_roc`: metrics from the
  validation set baked into `model_meta.json`. Fields may be `null` if the
  metadata file did not include them.
- `model_path`: server-side path to the loaded model file.

### curl example

```bash
curl http://localhost/model/info
```

---

## Common error format

All non-2xx responses use the same shape:

```json
{
  "detail": "Unsupported media type 'text/plain'. Allowed: image/jpeg, image/jpg, image/png, image/webp",
  "error_code": "UNSUPPORTED_MEDIA_TYPE"
}
```

For 429 responses the headers include:

```
Retry-After: 60
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1717593600
```

---

## Versioning

The API is versioned via `model_version` in responses, not via a path prefix.
A breaking change to request/response schemas would introduce a `/v2/` prefix
and we would maintain `/predict` and `/v2/predict` in parallel for at least
one minor release.
