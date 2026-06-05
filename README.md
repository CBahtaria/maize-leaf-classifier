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

Works on Linux, macOS, and Windows (Docker Desktop with Linux containers):

```bash
git clone https://github.com/CBahtaria/maize-leaf-classifier
cd maize-leaf-classifier
# Place model artifacts in model_artifacts/
docker compose -f docker/docker-compose.yml up -d
curl http://localhost/health
```

> **Windows:** use `copy .env.example .env` (CMD) or `Copy-Item .env.example .env` (PowerShell) instead of `cp`.

### Local development

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload          # API on :8000
cd frontend && npm ci && npm run dev   # PWA on :5173
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.main:app --reload          # API on :8000
Set-Location frontend; npm ci; npm run dev   # PWA on :5173
```

> **Note:** TensorFlow requires Python 3.9–3.12. Use `py -3.11` or install
> Python 3.11 from [python.org](https://www.python.org/downloads/) on Windows.

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

Zero-downtime deployments via nginx upstream switching.

**Linux / macOS:**
```bash
bash scripts/deploy.sh <git-sha>    # deploy new version
bash scripts/rollback.sh            # instant revert
```

**Windows (PowerShell — requires Docker Desktop):**
```powershell
.\scripts\deploy.ps1 -Tag <git-sha>   # deploy new version
.\scripts\rollback.ps1                # instant revert
```

See the [deployment guide](docs/report/deployment-guide.md) for full VPS setup.

## Colab inference demo

Serve the API from Google Colab with a public ngrok URL: open
`model/notebooks/ColabInference.ipynb` in Colab.

## Platform support

| Component | Linux | macOS | Windows |
|---|---|---|---|
| API (FastAPI + TFLite) | ✓ | ✓ | ✓ (Python 3.9–3.12) |
| Frontend (Vite + React) | ✓ | ✓ | ✓ |
| Docker deployment | ✓ | ✓ | ✓ (Docker Desktop) |
| Blue-green deploy | `deploy.sh` | `deploy.sh` | `deploy.ps1` |
| CI (GitHub Actions) | ubuntu-latest | — | windows-latest (frontend) |
| Production VPS | Linux only | — | — |

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
