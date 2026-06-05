# Model Card: MobileNetV2 Binary Maize Leaf Classifier v1.0.0

This model card follows the Google Model Cards reporting standard
(Mitchell et al., 2019).

---

## Model description

- **Architecture:** MobileNetV2 (ImageNet-pretrained), binary classification
  head (`GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) →
  Dense(1, sigmoid)`).
- **Format:** TFLite, INT8-quantised with representative-dataset activation
  calibration.
- **Input:** 224 × 224 × 3, RGB, raw `uint8` in `[0, 255]`. The correct
  architecture-specific `preprocess_input` is embedded inside the saved model
  as a `Lambda` layer (FIX-1 in the implementation report), so client code
  passes raw pixels and the model handles normalisation internally.
- **Output:** scalar `confidence ∈ [0, 1]` representing P(Diseased | image).
  A threshold of 0.5 maps to the binary label.
- **Size on disk:** ~3.5 MB.
- **Inference latency:** ~50 ms on a mid-range Android device (Cortex-A73
  class CPU), ~200 ms for the equivalent Keras `.h5` on a server CPU.
- **Training framework:** TensorFlow 2.15, Keras 3.
- **Author:** CBartaria.
- **Date:** 2026-06-05.
- **Version:** 1.0.0.
- **License:** MIT.

---

## Intended use

- **Primary use case:** Binary triage (Healthy / Diseased) of single
  maize leaves photographed by smallholder farmers using a smartphone camera.
- **Primary users:** Smallholder maize farmers in Eswatini and the broader
  Sub-Saharan African region.
- **Secondary users:** Agricultural extension officers conducting field
  visits, and researchers studying transfer-learning applications in
  agriculture.
- **Form factor:** Embedded in a Progressive Web App running on low-spec
  Android devices (1–2 GB RAM, Cortex-A53 / A73 class CPU). Also served via a
  FastAPI inference endpoint as the primary online path.

---

## Out-of-scope use

This model should **not** be used for:

- **Multi-class disease identification.** The model emits Healthy or Diseased
  only. It does not distinguish Common Rust from Northern Leaf Blight, Gray
  Leaf Spot, Cercospora, or other foliar diseases.
- **Disease severity grading.** No severity dimension is modelled.
- **Non-maize crops.** The model was trained exclusively on maize leaves and
  has no calibrated behaviour on tomato, bean, cassava, or other crop foliage.
- **Whole-plant or whole-field imagery.** The model expects a single leaf
  filling most of the frame. Aerial drone imagery or whole-plant photographs
  are out of distribution.
- **Insect or pest identification.** The model targets foliar disease
  symptoms only.
- **Sole-authority decisions.** The model is a decision support tool, not a
  diagnostic instrument. Treatment, intervention, and culling decisions
  should be informed by an agricultural officer.

---

## Training data

- **Primary dataset:** PlantVillage maize subset.
  - Source: PlantVillage open dataset.
  - Conditions: Lab — single leaf on a controlled background, even
    lighting, head-on framing.
  - Classes used: `Healthy`, `Cercospora Leaf Spot / Gray Leaf Spot`,
    `Common Rust`, `Northern Leaf Blight`. Diseased classes were merged into
    a single `Diseased` superclass for this binary model.
- **Secondary dataset (augmentation source):** "Maize in Field Dataset",
  Kaggle, contributor `hamishcrazeai`. Provides field-condition photographs
  used to reduce the lab-to-field distribution shift.
- **Auxiliary dataset (additional field samples):** "Maize, Beans, and
  Tomatoes Dataset for Africa", Kaggle, contributor `osutokaggle`. Used as a
  secondary source of African field photographs (maize subset only).
- **Train / val / test split:** Stratified by class, 70 / 15 / 15.
- **Class balance:** approximately 60% Diseased, 40% Healthy after merge.
  Class weights `w_c = N / (N_classes × N_c)` were used to correct.

---

## Factors

The model's performance varies along the following observable factors. Each
should be considered when interpreting a prediction in the field.

- **Image quality.** Blurred, out-of-focus, or low-light photographs reduce
  confidence and increase the false-negative rate.
- **Lighting conditions.** The training distribution is biased toward
  controlled lab lighting and bright daylight. Predictions in shade,
  artificial light, or sunset conditions are less reliable.
- **Leaf presentation angle.** The training distribution is centred on
  head-on photographs of a single leaf. Oblique angles, partial occlusion by
  other leaves, and overlapping leaves reduce accuracy.
- **Distance from leaf.** The leaf should occupy ~80% of the frame.
  Photographs that capture the whole plant or the whole field are out of
  distribution.
- **Background clutter.** Uniform backgrounds (sky, soil, the farmer's hand)
  yield more confident predictions than densely cluttered backgrounds
  (other plants, foliage at multiple depths).
- **Maize variety.** Training data covers a range of varieties but is not
  exhaustive. Less-common landraces in Eswatini may be slightly out of
  distribution.

---

## Metrics

All metrics are computed on the held-out 15% validation split of the merged
PlantVillage + field datasets.

| Metric | Value |
|---|---|
| Accuracy | ~0.94 |
| Sensitivity (Recall, Diseased) | ~0.92 |
| Specificity | ~0.95 |
| AUC-ROC | ~0.97 |
| F1 | ~0.93 |
| Threshold | 0.5 (default; tunable per deployment) |
| TFLite size on disk | ~3.5 MB |
| CPU inference latency | ~50 ms on Cortex-A73 class Android |

**Note:** these figures are estimates from internal evaluation on the
PlantVillage validation set with the augmentation pipeline described in the
full report. Real-world field performance must be validated with labelled
farmer-collected data before any operational dependency on the numbers above.

---

## Ethical considerations

- **Lab-to-field distribution shift.** The PlantVillage primary dataset is
  captured in lab conditions. Field photographs taken by farmers will have
  different lighting, framing, background, and image quality. Real-world
  accuracy is expected to be lower than the validation accuracy reported
  above. Deployment claims must avoid implying validation accuracy translates
  directly to field accuracy.
- **Asymmetric cost of errors.** A **false negative** (diseased leaf
  classified as healthy) is materially worse for the farmer than a **false
  positive** (healthy leaf classified as diseased): a missed disease can
  spread through the field before the next inspection, while a false alarm
  costs at most one additional manual check. Deployments serving real
  farmers should consider raising the decision threshold (e.g. from 0.5 to
  0.4 on P(Diseased)) to bias toward sensitivity at the cost of specificity.
- **Decision-support framing.** The model should be presented to farmers as
  a decision-support tool, never as a sole authority. UI strings emphasise
  "may be sick — check other leaves and speak to an agricultural officer"
  rather than "your plant has disease X."
- **Privacy.** Photographs frequently contain GPS EXIF metadata that
  pinpoints a farmer's land. The API layer strips all metadata before
  inference; photographs are not persisted server-side. The offline path
  runs entirely on-device.
- **Equitable access.** The PWA installs without an app store, runs on
  low-spec Android, and is functional offline. These are deliberate design
  choices intended to lower the access threshold for the target user
  population.

---

## Caveats and recommendations

- Accuracy figures are **estimates** from the PlantVillage validation set.
  Production deployments should validate against labelled field data from
  the deployment region before reporting accuracy externally.
- The model's binary output is **not** a substitute for disease-specific
  treatment guidance. Future versions plan to add multi-class
  identification, severity grading, and Grad-CAM visualisation to make the
  prediction more actionable.
- Re-evaluation against a new validation set is recommended any time the
  upstream `tf.keras.applications.MobileNetV2` implementation changes, in
  case the dynamic fine-tune-layer computation (FIX-5) selects a different
  subnetwork.
- Distribution-shift detection (e.g. ODIN, energy-based scoring) is not
  implemented in v1. Out-of-distribution inputs (non-leaf photos) will
  produce confident-but-meaningless predictions. Users should be aware that
  the model has no native "I don't know" output.
