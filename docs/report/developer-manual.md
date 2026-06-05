# Developer Manual

A practical guide for developers contributing to the maize-leaf-classifier
project. Covers local setup, architecture, ML pipeline, API development,
frontend development, testing, deployment, Colab integration, the seven
research-paper fixes, environment variables, and troubleshooting.

---

## 1. Prerequisites

| Tool | Required version | Notes |
|---|---|---|
| Python | 3.11.x (not 3.12+) | TF 2.15 wheels are not yet published for 3.12 |
| Node.js | 20.x or newer | For the Vite + React frontend |
| Docker | 24.x or newer | With the `compose` plugin |
| Git | 2.30+ | Worktrees feature used for branch isolation |
| GNU coreutils, bash | any modern | Required by `scripts/deploy.sh` |

Optional:

- `kaggle` CLI for dataset downloads.
- `ngrok` account (for the Colab inference demo).

---

## 2. Quick start

### 2.1 One-command Docker dev environment (recommended, any OS)

No local Python or Node required — both services run in containers.

```bash
# Linux / macOS
git clone <repo> && cd maize-leaf-classifier
docker compose -f docker/docker-compose.dev.yml up --build

# Windows PowerShell
git clone <repo>; cd maize-leaf-classifier
docker compose -f docker/docker-compose.dev.yml up --build
```

- API: `http://localhost:8000` (auto-reloads on `api/` and `model/` changes)
- Frontend: `http://localhost:5173` (Vite HMR, proxies `/predict*` to API
  container via Docker DNS `api:8000`)

The `frontend_node_modules` named volume keeps `node_modules` inside the
container so Windows host paths never shadow it.

### 2.2 Native setup (Python 3.11 + Node 20)

```bash
git clone <repo> && cd maize-leaf-classifier
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn api.main:app --reload
# In another terminal:
cd frontend && npm install && npm run dev
```

The API serves on `http://localhost:8000`, the Vite dev server serves on
`http://localhost:5173` and proxies `/predict*` to `:8000`.

---

## 3. Project architecture

```
maize-leaf-classifier/
├── api/                    FastAPI service
│   ├── main.py             app factory + lifespan + CORS
│   ├── config.py           pydantic Settings
│   ├── schemas.py          request/response models
│   ├── dependencies.py     model loader + version metadata
│   ├── routes/             one file per endpoint family
│   │   ├── predict.py
│   │   ├── batch.py
│   │   └── health.py
│   └── middleware/         security middlewares
│       ├── validation.py   MIME + size + PIL verify + EXIF strip
│       └── rate_limit.py   slowapi limiter
├── model/                  ML pipeline
│   ├── config.py           hyperparameters
│   ├── preprocess.py       tf.data pipeline + preprocess_input
│   ├── augmentation.py     Keras 3 GPU augmentation layers
│   ├── architectures.py    5 model families + dynamic fine-tune
│   ├── callbacks.py        LinearWarmupCallback + early stop
│   ├── train.py            two-phase training entry point
│   ├── evaluate.py         metrics + confusion matrix
│   ├── export.py           TFLite INT8 conversion
│   ├── predict.py          single-image inference helper
│   └── notebooks/
│       ├── MaizeDiseaseClassifier_v1.ipynb    Training
│       └── ColabInference.ipynb               Demo
├── frontend/               React 18 + Vite PWA
│   ├── src/
│   │   ├── components/     CameraCapture, ResultCard, etc.
│   │   ├── hooks/          useCamera, usePredict
│   │   ├── lib/            api.js, tfjs-inference.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   │   ├── manifest.json
│   │   └── sw.js           service worker
│   └── vite.config.js
├── docker/
│   ├── docker-compose.yml  blue/green API + nginx + frontend builder
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   ├── nginx.conf          static main config
│   └── conf.d/
│       └── active-upstream.conf   mutated by deploy.sh
├── scripts/
│   ├── deploy.sh           blue-green deploy
│   ├── rollback.sh         instant rollback
│   ├── convert_tfjs.py     Keras → TF.js export
│   └── generate_icons.py   PWA icon generator
├── tests/
│   ├── conftest.py         fixtures (mock model, etc.)
│   ├── unit/
│   └── integration/
├── docs/                   this directory
├── model_artifacts/        (gitignored) trained models
└── .github/workflows/      CI/CD
```

---

## 4. Development environment

### 4.1 API development

```bash
uvicorn api.main:app --reload --port 8000
```

- `--reload` watches `api/` for changes and restarts.
- The mock-model `conftest.py` is **not** active in `uvicorn` — you need a
  real `.tflite` in `model_artifacts/` or set `MODEL_PATH` to a test fixture.

### 4.2 Frontend development

```bash
cd frontend
npm ci
npm run dev
```

Vite dev server runs on `:5173`. `vite.config.js` proxies `/predict`,
`/predict/batch`, `/health`, and `/model/info` to `http://localhost:8000`.

### 4.3 Full-stack workflow

Run both servers in separate terminals. Open `http://localhost:5173` in
Chrome. The dev server hot-reloads on save; the API server hot-reloads
on save. End-to-end testing of camera → API → result is possible without
Docker.

### 4.4 Pre-commit checks

Run before opening a PR:

```bash
ruff check api/ model/ tests/
ruff format --check api/ model/ tests/
mypy api/ model/
pytest tests/ -v
cd frontend && npm run lint
```

---

## 5. ML pipeline guide

### 5.1 Dataset structure

Organise raw images into per-class subdirectories. The training pipeline
auto-detects subdirectory names and merges everything except `healthy` into
the `Diseased` superclass.

```
data/
├── healthy/
│   ├── img001.jpg
│   └── img002.jpg
├── common_rust/
│   └── *.jpg
├── northern_leaf_blight/
│   └── *.jpg
├── gray_leaf_spot/
│   └── *.jpg
└── cercospora_leaf_spot/
    └── *.jpg
```

The disease-name detection is governed by `DISEASE_LABELS` in
`model/config.py`. Add new directory names to this set if you incorporate a
new dataset whose folders use unfamiliar naming.

### 5.2 Train

```bash
python -m model.train \
    --arch mobilenetv2 \
    --data data/ \
    --output model_artifacts/
```

Available `--arch`: `mobilenetv2`, `xception`, `inceptionv3`, `vgg16`,
`resnet50`.

Output artifacts:

- `<arch>_best.keras` — best checkpoint by validation AUC.
- `<arch>_best.tflite` — INT8-quantised export (auto).
- `<arch>_meta.json` — version, metrics, training config.

### 5.3 Evaluate

```bash
python -m model.evaluate \
    --model model_artifacts/mobilenetv2_best.keras \
    --data data/
```

Prints accuracy, precision, recall (sensitivity), specificity, F1, AUC, and
a confusion matrix.

### 5.4 Export TFLite manually

`train.py` exports TFLite automatically. To re-export from an existing
`.keras`:

```bash
python -m model.export \
    --keras-model model_artifacts/mobilenetv2_best.keras \
    --output model_artifacts/mobilenetv2_best.tflite \
    --data data/
```

### 5.5 Export TF.js

```bash
python scripts/convert_tfjs.py \
    --keras-model model_artifacts/mobilenetv2_best.keras \
    --output frontend/public/tfjs/
```

The Vite build copies `public/tfjs/` into the PWA bundle.

---

## 6. API development

### 6.1 Add a new endpoint

1. Create a new module under `api/routes/` (e.g. `api/routes/explain.py`).
2. Define a router: `router = APIRouter()`.
3. Register the router in `api/main.py`:

```python
from api.routes import explain
app.include_router(explain.router)
```

4. Add request/response schemas to `api/schemas.py`.
5. If the endpoint accepts file uploads, reuse `validate_image_file` from
   `api/middleware/validation.py`.

### 6.2 Local testing

```bash
pytest tests/integration/ -v
```

The `conftest.py` patches the TFLite loader to return a deterministic mock
interpreter, so tests do not depend on a real `model_artifacts/` directory.

### 6.3 Security middleware

The validation middleware is **always active** on file-upload endpoints —
you cannot bypass it. Tests must use real (small) image bytes, not arbitrary
strings. The `conftest.py` provides a fixture `tiny_jpeg_bytes()` for this.

---

## 7. Frontend development

### 7.1 Code layout

- `frontend/src/components/` — presentational React components.
- `frontend/src/hooks/` — reusable React hooks.
- `frontend/src/lib/api.js` — `fetch` wrapper for the FastAPI endpoints.
- `frontend/src/lib/tfjs-inference.js` — TF.js model loader and runner for
  the offline path.

### 7.2 PWA build and preview

```bash
cd frontend
npm run build      # outputs to dist/
npm run preview    # serves dist/ on :4173
```

Audit the PWA quality with Chrome DevTools → Lighthouse → "Progressive Web
App" category. Target score ≥ 90.

### 7.2.1 FAO-inspired design system

The stylesheet (`frontend/src/styles/main.css`) follows the FAO
(fao.org) institutional design language:

| Token | Value | Usage |
|---|---|---|
| `--fao-blue` | `#0059AA` | Primary action color, header gradient |
| `--fao-blue-dark` | `#003F87` | Header gradient start, hover state |
| `--healthy` | `#1B7A3E` | Healthy diagnosis card header + fill |
| `--diseased` | `#C0392B` | Diseased diagnosis card header + fill |

Key component classes:
- `.diagnosis-card--healthy / --diseased` — full-bleed colored header
- `.upload-zone` — dashed-border dropzone with hover states
- `.confidence-section` / `.confidence-track` / `.confidence-fill--*` — animated bar
- `.history-table` / `.history-row` — 4-column grid (badge, %, time, offline)

All color decisions are via CSS variables so a theme swap requires only
editing the `:root {}` block in `main.css`.

### 7.3 Offline testing

1. `npm run build && npm run preview`.
2. Open in Chrome, ensure service worker registers.
3. DevTools → Application → Service Workers → check "Offline".
4. DevTools → Network → set throttling to "Offline".
5. Reload — the app shell loads from cache, predictions route through
   `tfjs-inference.js`.

---

## 8. Testing

### 8.1 Unit tests

```bash
pytest tests/unit/ -v
```

Key tests:

- `test_preprocess.py::test_mobilenetv2_preprocess_range` — **critical
  regression test for FIX-1.** Must pass before any deploy.
- `test_preprocess.py::test_class_weight_formula` — verifies the
  `w_c = N / (N_classes × N_c)` formula.
- `test_augmentation.py::test_augmentation_shape` — augmentation preserves
  tensor shape.
- `test_callbacks.py::test_linear_warmup` — warmup ramps LR linearly and
  then disables itself.

### 8.2 Integration tests

```bash
pytest tests/integration/ -v
```

These exercise the FastAPI app with a mocked model. They cover the full
validation chain (415, 413, 400) and the rate-limit response (429).

### 8.3 Coverage

```bash
pytest --cov=api --cov=model --cov-report=term-missing
```

Target: ≥ 80% line coverage on `api/middleware/` and `model/preprocess.py`.

---

## 9. Blue-green deployment (developer view)

### 9.1 Mental model

Two API containers (`api-blue`, `api-green`) run in parallel. nginx routes
all traffic to the active one via
`docker/conf.d/active-upstream.conf`, which `scripts/deploy.sh` mutates.

### 9.2 Deploy

```bash
bash scripts/deploy.sh <git-sha>
```

The script pulls the new image into the inactive slot, polls health, then
flips nginx. Failed health → no flip; production stays on the old slot.

### 9.3 Rollback

```bash
# Linux / macOS
bash scripts/rollback.sh

# Windows PowerShell
.\scripts\rollback.ps1
```

Flips back immediately. No health check.

### 9.3.1 Windows PowerShell scripts

All shell scripts have PowerShell equivalents:

| bash | PowerShell |
|---|---|
| `scripts/deploy.sh` | `scripts/deploy.ps1` |
| `scripts/rollback.sh` | `scripts/rollback.ps1` |

The PowerShell variants write nginx upstream config with LF line endings
using `[System.IO.File]::WriteAllText` — required because nginx on Linux
rejects files with Windows CRLF endings.

### 9.4 GitHub Secrets

Set the following in repo Settings → Secrets and variables → Actions:

| Secret | Description |
|---|---|
| `VPS_HOST` | Production server hostname or IP |
| `VPS_USER` | SSH user with deploy permissions |
| `VPS_SSH_KEY` | Private SSH key (matched against authorized_keys on VPS) |

Full deployment-side details are in `docs/report/deployment-guide.md`.

---

## 10. Colab integration

### 10.1 Training notebook

`model/notebooks/MaizeDiseaseClassifier_v1.ipynb`:

1. Open in [Colab](https://colab.research.google.com/) (file → upload notebook,
   or open from GitHub).
2. Mount Google Drive and place training data at `MyDrive/maize-data/`.
3. Place output target at `MyDrive/maize-model/`.
4. Run cells top to bottom. End-to-end runtime: ~2–3 hours on a T4 GPU.

### 10.2 Inference demo notebook

`model/notebooks/ColabInference.ipynb`:

1. Open in Colab.
2. Place trained `mobilenetv2_best.tflite` and `mobilenetv2_meta.json` in
   `MyDrive/maize-model/`.
3. Add your ngrok auth token to the cell that calls `ngrok.set_auth_token(...)`.
4. Run all cells. The final cell prints a public URL and renders a QR code
   that you can scan with a phone to reach the API.

Both notebooks expect artifacts in `MyDrive/maize-model/`. This convention
keeps the notebooks portable across Google accounts.

---

## 11. Research inconsistencies implemented

| Fix | Paper says | What's wrong | What was fixed | Where in code |
|---|---|---|---|---|
| FIX-1 | `/255` for all architectures | MobileNetV2 expects `[-1, 1]`, VGG16 expects ImageNet mean-subtracted BGR. 5–20% accuracy penalty | Embedded `preprocess_input` as a `Lambda` layer inside the saved model | `model/preprocess.py::get_preprocess_fn`, `model/architectures.py::build_model` |
| FIX-2 | `ImageDataGenerator` | Deprecated; CPU bottleneck; no `tf.data` integration | Replaced with `tf.data` + Keras 3 GPU augmentation layers | `model/preprocess.py`, `model/augmentation.py` |
| FIX-3 | "Deploy via TFLite" | No TFLite export step specified | Full INT8-quantised TFLite export with representative dataset | `model/export.py`, `model/train.py` |
| FIX-4 | "Linear LR warmup" | Keras has no built-in linear warmup | Custom `LinearWarmupCallback` that disables itself after warmup | `model/callbacks.py::LinearWarmupCallback` |
| FIX-5 | Hardcoded layer indices (e.g. `layer_index=100`) | Indices change across TF versions | Compute unfreeze point dynamically from the top | `model/architectures.py::fine_tune_model`, `model/config.py::FINE_TUNE_LAYERS` |
| FIX-6 | `colad.research.google.com` (typo) | DNS NXDOMAIN; reproducibility broken | All references corrected to `colab.research.google.com` | README, developer manual, notebook metadata |
| FIX-7 | "Mobile inference: 50 ms (.h5)" | Benchmark loads `.h5`, not `.tflite`; misleading | Benchmarks now use TFLite; two distinct numbers quoted | `model/predict.py`, full-report.md §12 |

---

## 12. Environment variables reference

All settings are loaded by `api/config.py` via `pydantic-settings`. Defaults
in parentheses.

| Variable | Type | Default | Description |
|---|---|---|---|
| `MODEL_PATH` | str | `model_artifacts/mobilenetv2_best.tflite` | Absolute or relative path to the TFLite model. |
| `MODEL_META_PATH` | str | `model_artifacts/mobilenetv2_meta.json` | Path to the metadata JSON. |
| `ALLOWED_ORIGINS` | list[str] (JSON) | `["http://localhost","http://localhost:5173"]` | CORS allowed origins. |
| `LOG_LEVEL` | str | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `MAX_UPLOAD_BYTES` | int | `10485760` | Hard cap on uploaded file size, in bytes (10 MB). |
| `MAX_BATCH_SIZE` | int | `10` | Maximum images per `/predict/batch` request. |
| `DOMAIN` | str | `localhost` | Public domain (used by Docker Compose for ALLOWED_ORIGINS expansion). |
| `API_TAG` | str | `latest` | Container image tag (consumed by `docker-compose.yml`). |
| `GITHUB_REPOSITORY` | str | `local` | Used to build the ghcr.io image reference. |

The repository ships `.env.example` with these defaults. Copy to `.env`
before running Docker Compose.

---

## 13. Troubleshooting

### `ImportError: tensorflow not found`

You are on Python 3.12 or newer. TensorFlow 2.15 wheels are not yet
published for Python 3.12. Recreate the venv with Python 3.11:

```bash
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### `503 MODEL_NOT_LOADED`

The API is up but failed to load the model.

- Check `MODEL_PATH` in `.env` resolves to a real file inside the container.
- Inside Docker, the path is `/app/model_artifacts/<file>.tflite`.
- Verify the volume mount: `docker compose -f docker/docker-compose.yml
  exec api-blue ls -la /app/model_artifacts/`.

### `CORS error in browser`

The Vite dev server origin (`http://localhost:5173`) is not in
`ALLOWED_ORIGINS`. Add it in `.env`:

```env
ALLOWED_ORIGINS=["http://localhost","http://localhost:5173"]
```

Restart the API.

### `429 in tests`

The module-level `limiter` singleton in `api/middleware/rate_limit.py`
persists its in-memory counter across the entire pytest session. The
`conftest.py` `reset_rate_limiter` autouse fixture clears `limiter._storage`
before every test so each test starts from zero. If you add a new test file
that is **not** covered by the top-level `conftest.py`, import and apply the
fixture explicitly, or move the test to the `tests/` tree.

### `Docker build fails on ARM`

TensorFlow 2.15 has limited ARM wheel support. Build with the AMD64
platform flag explicitly:

```bash
docker compose -f docker/docker-compose.yml build --platform=linux/amd64
```

If running on Apple Silicon for local development, expect emulation
overhead (~3× slower builds).

### `tflite-runtime missing on dev machine`

`requirements.txt` pins `tensorflow==2.15.*` which includes the `tflite`
interpreter as `tf.lite.Interpreter`. The lightweight `tflite-runtime`
package is only used inside the production Docker image. Use the full
TensorFlow on dev.

### `Service worker not updating`

Service workers cache aggressively. In Chrome DevTools → Application →
Service Workers, check "Update on reload" and click "Unregister", then
reload.

### `ngrok URL changes every Colab restart`

ngrok free tier issues a new random URL each session. For a stable URL
during a longer demo, use a paid ngrok plan and configure a reserved
domain in the `ColabInference.ipynb` ngrok cell.
