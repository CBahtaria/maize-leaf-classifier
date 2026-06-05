# Maize Leaf Disease Classifier — Design Specification

**Date:** 2026-06-05
**Status:** Approved (implementation in progress)
**Author:** Project team
**Repo:** maize-leaf-classifier
**Branch:** feature/initial-implementation

---

## 1. Problem Statement

Smallholder maize farmers in Sub-Saharan Africa (SSA) — and specifically Eswatini, the
primary deployment target — face significant yield losses from common maize foliar
diseases (Northern Leaf Blight, Common Rust, Gray Leaf Spot, Cercospora). Early detection
allows targeted intervention before disease spreads through a field, but expert
agricultural extension is scarce and visits are infrequent.

Existing crop-disease applications fail this user base for three structural reasons:

1. **Device constraints.** The dominant smartphone profile in rural SSA is a low-spec
   Android device (1–2 GB RAM, ARM Cortex-A53, limited storage). Native apps shipping
   100+ MB of model weights crash on installation or are removed to free space.
2. **App store friction.** Many farmers do not have a Google account configured, do not
   trust the Play Store install flow, and lack the data budget to download a 100 MB
   `.apk`. Distribution via the Play Store is a real barrier, not a theoretical one.
3. **Intermittent connectivity.** Field conditions vary from no signal at all to 2G
   only. An online-only architecture is unusable in the field. An offline-only
   architecture forfeits the accuracy gains of larger server-side models.

The system must therefore work on the cheapest Android devices available, install
without an app store, and remain functional whether the farmer is online or offline.

## 2. Architecture Decisions

### 2.1 Progressive Web App — not React Native

We deliberately chose a **PWA** over React Native, Flutter, or a native Android app.

**Why PWA wins for this audience:**

- **No app store.** PWAs install via "Add to Home Screen" in Chrome or Safari. No
  Google account, no Play Store, no `.apk` sideloading. A farmer follows a SMS link,
  taps Install, and the icon appears on their home screen within seconds.
- **Tiny initial download.** The compiled PWA bundle is ~150 KB gzipped. The TF.js
  model is fetched only when offline mode is first activated, so most users never pay
  the cost.
- **Updates are instant.** A service worker push updates all users on next launch.
  No "85% of users on version 1.0" tail to drag behind us.
- **One codebase.** React + Vite renders identically on Android and iOS. We do not
  maintain parallel native code.

**What we give up:** Background sync, push notifications, true filesystem access. We
do not need these for v1.

### 2.2 API-primary with TF.js offline fallback

The system has two inference paths and chooses automatically based on connectivity.

- **Online (preferred):** image → POST `/predict` → FastAPI → TFLite INT8 model
  → JSON response. The server-side model is the same MobileNetV2 trained on the same
  data; serving it via the API isolates the heavy lifting from the device.
- **Offline (fallback):** if the API is unreachable, the PWA loads a cached TF.js
  copy of the model and runs inference in the browser. Accuracy parity with the
  server model is maintained because both are exported from the same Keras checkpoint.

This split lets us deliver a high-accuracy answer on the common path while ensuring
the app is never broken by a flaky connection.

### 2.3 Preprocessing embedded in the saved model (FIX-1)

The original research paper normalised inputs by dividing by 255 for all
architectures. This is wrong for `MobileNetV2` (which expects `[-1, 1]`) and for
`VGG16` (which expects ImageNet mean-subtracted BGR). We embed the correct
architecture-specific `preprocess_input` as a `Lambda` layer inside the saved
`.keras` and `.tflite` models. Client code passes raw `uint8` pixels in `[0, 255]`
and the model handles normalisation. This eliminates an entire class of
"preprocessing drift" bugs between training and serving.

### 2.4 TFLite INT8 export with representative dataset (FIX-3)

We export the full Keras `.h5` model (~14 MB) to TFLite with INT8 weight quantisation
plus a representative dataset for activation calibration. The result is ~3.5 MB
on disk and 2–4× faster CPU inference than the Keras model. INT8 quantisation
preserves accuracy within ~1% for MobileNetV2 on the maize classification task in our
internal evaluation, while making the model usable on low-spec devices and minimising
the API container image size.

### 2.5 Blue-green deployment on a single VPS

Production is a single low-spec VPS (2 vCPU, 2 GB RAM, Ubuntu 22.04). We run **both**
API containers (`api-blue`, `api-green`) simultaneously behind nginx. The active
slot is determined by a file at `docker/conf.d/active-upstream.conf` which is
rewritten by `scripts/deploy.sh`. Deployment:

1. New image is pulled into the inactive slot.
2. Health is polled directly against the inactive container.
3. nginx upstream is rewritten + reloaded — zero dropped connections.
4. Old slot stays running. `scripts/rollback.sh` switches back in ~1 second.

This gives us atomic, zero-downtime deploys and an instant rollback path without
the complexity of Kubernetes, Nomad, or a load balancer. For a project serving
demonstration traffic from a single region, this is the right trade-off.

### 2.6 Colab inference for demos

A separate `ColabInference.ipynb` runs the FastAPI app inside a Colab session,
exposed via `ngrok` for a public URL. This is **demo-only** — it lets stakeholders
try the app from a phone in 60 seconds without us paying for a VPS or hosting bill.
Session limits and ngrok bandwidth caps make Colab unsuitable for production.

### 2.7 Security model

File-upload endpoints are the highest-risk surface. Defences (in order):

1. **MIME whitelist.** Only `image/jpeg`, `image/jpg`, `image/png`, `image/webp` are
   accepted. The `Content-Type` header is inspected — file extensions are ignored.
2. **Hard size cap.** Reads at most `max_bytes + 1` to detect oversized uploads
   without loading them entirely into memory. Default cap: 10 MB.
3. **PIL `verify()`.** Detects corrupt headers, truncated files, and
   decompression bombs before we attempt to decode.
4. **PIL `load()`.** Fully decodes the image after re-open (required because
   `verify()` exhausts the internal buffer).
5. **EXIF stripping.** All metadata — including GPS coordinates from field photos —
   is dropped during RGB conversion. Farmer location is never sent to the model.
6. **Dimension cap.** Images larger than 10 000 px per side are rejected as
   decompression-bomb risks.
7. **Rate limiting.** `slowapi` enforces 20 req/min per IP on `/predict` and
   `/predict/batch`, returning HTTP 429 with `Retry-After: 60`.
8. **Non-root Docker user.** Both API containers run as a non-root user; the
   model artifacts volume is mounted read-only.

## 3. Component Diagram

```
+--------------------+              +------------------+
|  Farmer's Phone    |              |  Stakeholder     |
|  Android PWA       |              |  Demo (Colab)    |
|  (React + Vite)    |              |  ngrok URL       |
+----------+---------+              +---------+--------+
           |                                  |
           | HTTPS                            | HTTPS
           v                                  v
+--------------------+              +------------------+
|       nginx        |              |  FastAPI in      |
|  reverse proxy +   |              |  Colab notebook  |
|  upstream switch   |              +------------------+
+----+----------+----+
     |          |
     | active   | (warm standby)
     v          v
+--------+  +--------+
| api-   |  | api-   |
| blue   |  | green  |
| FastAPI|  | FastAPI|
+---+----+  +---+----+
    |           |
    +-----+-----+
          |
          v
   +-------------+
   | model_      |
   | artifacts/  |
   | (read-only) |
   |  *.tflite   |
   |  meta.json  |
   +-------------+

Offline path (no network):
+--------------------+
|  Farmer's Phone    |
|  Service worker -> |
|  TF.js model       |
|  (cached in        |
|  IndexedDB)        |
+--------------------+

Training path (offline, not part of runtime):
+--------------------+    +---------------------+
|  Colab notebook    | -> |  model_artifacts/   |
|  Phase 1 + Phase 2 |    |  *.keras, *.tflite, |
|  → TFLite export   |    |  meta.json, tfjs/   |
+--------------------+    +---------------------+
```

## 4. Out of Scope

- Multi-class disease identification (Common Rust vs Northern Leaf Blight vs Gray
  Leaf Spot). v1 is binary triage only.
- Severity grading.
- Crops other than maize.
- A user account / login system. The PWA is anonymous.
- Native push notifications.
- Background sync of offline predictions.
- Localisation to siSwati or Swahili. UI is English-only in v1.
