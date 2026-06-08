#!/usr/bin/env bash
# release-model.sh — Upload trained model artifacts to GitHub Releases.
#
# Usage:
#   bash scripts/release-model.sh <path/to/inceptionv3_best.tflite> [<path/to/inceptionv3_meta.json>]
#
# After running:
#   1. Copy the printed MODEL_DOWNLOAD_URL
#   2. In Render dashboard → maizescan-api → Environment → set MODEL_DOWNLOAD_URL
#   3. Trigger a manual redeploy in Render (the API will download the model on startup)
set -euo pipefail

TFLITE="${1:-}"
META="${2:-$(dirname "${TFLITE:-}")/inceptionv3_meta.json}"
REPO="CBahtaria/maize-leaf-classifier"
TAG="model-v1.0.0"
TITLE="InceptionV3 binary classifier v1.0.0"

die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }

# ── Preflight checks ────────────────────────────────────────────────────────
[[ -n "$TFLITE" ]]       || die "Usage: bash $0 <path/to/inceptionv3_best.tflite>"
[[ -f "$TFLITE" ]]       || die "TFLite file not found: $TFLITE"
command -v gh >/dev/null || die "'gh' CLI not installed — install from https://cli.github.com"
gh auth status >/dev/null 2>&1 || die "Not authenticated with GitHub — run: gh auth login"

# Meta JSON is optional — skip if not present
UPLOAD_META=false
if [[ -f "$META" ]]; then
    UPLOAD_META=true
else
    warn "Meta JSON not found at ${META} — will upload TFLite only"
fi

SIZE_MB=$(du -m "$TFLITE" | cut -f1)
info "TFLite: ${TFLITE}  (${SIZE_MB} MB)"
$UPLOAD_META && info "Meta:   ${META}"

# ── Delete existing tag/release if present ──────────────────────────────────
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    warn "Release ${TAG} already exists — deleting and recreating"
    gh release delete "$TAG" --repo "$REPO" --yes --cleanup-tag 2>/dev/null || true
    sleep 2
fi

# ── Create release ───────────────────────────────────────────────────────────
info "Creating GitHub Release ${TAG}..."

NOTES="## MaizeScan Model — InceptionV3 Binary Classifier

| Property | Value |
|----------|-------|
| Architecture | InceptionV3 (224×224 input) |
| Task | Binary: Healthy vs. Diseased maize leaf |
| Export format | INT8 quantized TFLite |
| Training data | Dual-dataset (Maize in Field + PlantVillage) |

### Files
- \`inceptionv3_best.tflite\` — INT8 quantized model (~4 MB) for the FastAPI backend
- \`inceptionv3_meta.json\` — metrics and version metadata

### Deploying
Set \`MODEL_DOWNLOAD_URL\` in your Render environment to the direct download URL of
\`inceptionv3_best.tflite\` printed below. The API downloads the model at startup."

if $UPLOAD_META; then
    gh release create "$TAG" \
        --repo "$REPO" \
        --title "$TITLE" \
        --notes "$NOTES" \
        "$TFLITE#inceptionv3_best.tflite" \
        "$META#inceptionv3_meta.json"
else
    gh release create "$TAG" \
        --repo "$REPO" \
        --title "$TITLE" \
        --notes "$NOTES" \
        "$TFLITE#inceptionv3_best.tflite"
fi

ok "Release created: https://github.com/${REPO}/releases/tag/${TAG}"

# ── Print the direct download URL ───────────────────────────────────────────
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/inceptionv3_best.tflite"

printf '\n'
printf '\033[1;32m╔══════════════════════════════════════════════════════════════╗\033[0m\n'
printf '\033[1;32m║  MODEL_DOWNLOAD_URL (copy this into Render env vars)         ║\033[0m\n'
printf '\033[1;32m╚══════════════════════════════════════════════════════════════╝\033[0m\n'
printf '\n  %s\n\n' "$DOWNLOAD_URL"
printf 'Next steps:\n'
printf '  1. Render dashboard → maizescan-api → Environment\n'
printf '     MODEL_DOWNLOAD_URL = %s\n' "$DOWNLOAD_URL"
printf '  2. Click "Save Changes" → "Manual Deploy" → "Deploy latest commit"\n'
printf '  3. Check logs: API will print "Download complete" on startup\n'
printf '  4. Verify: curl https://maizescan-api.onrender.com/health\n'
