# Maize Leaf Disease Classifier

Binary CNN classifier (MobileNetV2) for healthy vs. diseased maize leaf
detection, targeting smallholder farmers in Sub-Saharan Africa on low-spec
Android devices.

## What it does

- Farmers photograph a maize leaf with their smartphone.
- The app returns "Healthy" or "Diseased" in < 500 ms.
- Works offline via a TF.js model cached in the browser.
- Installable as a PWA — no app store required.

## Quick start

### Run with Docker

```bash
git clone <repo> && cd maize-leaf-classifier
cp .env.example .env
# Place model artifacts in model_artifacts/
docker compose -f docker/docker-compose.yml up -d
curl http://localhost/health
```

### Local development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload          # API on :8000
cd frontend && npm ci && npm run dev   # PWA on :5173
```

### Train a model

1. Download datasets from Kaggle:

   ```bash
   kaggle datasets download hamishcrazeai/maize-in-field-dataset
   kaggle datasets download osutokaggle/maize-beans-and-tomatoes-dataset-for-africa
   ```

2. Organise into `data/healthy/` and `data/<disease>/` directories.
3. Run training:

   ```bash
   python -m model.train --arch mobilenetv2 --data data/ --output model_artifacts/
   ```

Or use the Colab notebook: `model/notebooks/MaizeDiseaseClassifier_v1.ipynb`.

## Blue-green deployment

Zero-downtime deployments via nginx upstream switching:

```bash
# Deploy new version
bash scripts/deploy.sh <git-sha>

# Rollback if needed
bash scripts/rollback.sh
```

See the [deployment guide](docs/report/deployment-guide.md) for full setup.

## Colab inference demo

Serve the API from Google Colab with a public ngrok URL: open
`model/notebooks/ColabInference.ipynb` in Colab.

## Documentation

| Document | Description |
|---|---|
| [Full Report](docs/report/full-report.md) | Technical report including research inconsistency analysis |
| [API Reference](docs/report/api-reference.md) | All API endpoints with examples |
| [Developer Manual](docs/report/developer-manual.md) | Setup, architecture, contributing |
| [User Manual](docs/report/user-manual.md) | End-user guide for farmers |
| [Deployment Guide](docs/report/deployment-guide.md) | Docker + blue-green deployment |
| [Model Card](docs/report/model-cards/mobilenetv2-binary-v1.md) | Model metadata and ethics |

## Research fixes applied

This implementation corrects 7 technical inconsistencies found in the original
research paper:

| Fix | Issue | Impact |
|---|---|---|
| FIX-1 | `/255` normalisation → architecture-specific `preprocess_input` | 5–20% accuracy penalty |
| FIX-2 | `ImageDataGenerator` → `tf.data` + GPU augmentation | CPU→GPU training |
| FIX-3 | Missing TFLite export | 14 MB → 3.5 MB, 4× faster |
| FIX-4 | Missing `LinearWarmupCallback` | Training protocol correctness |
| FIX-5 | Hardcoded layer indices → dynamic calculation | TF version compatibility |
| FIX-6 | Colab URL typo "colad" → "colab" | Reproducibility |
| FIX-7 | `.h5` benchmark → TFLite benchmark | Deployment accuracy |

## License

MIT
