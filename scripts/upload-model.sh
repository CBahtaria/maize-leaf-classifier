#!/usr/bin/env bash
# upload-model.sh — copy model artifacts to VPS and restart API containers
# Usage: bash scripts/upload-model.sh user@host
set -euo pipefail

TARGET="${1:-}"
LOCAL_DIR="model_artifacts"
REMOTE_DIR="/opt/maize-leaf-classifier/model_artifacts"
TFLITE="inceptionv3_best.tflite"
META="inceptionv3_meta.json"

die() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }

[[ -n "$TARGET" ]] || die "Usage: bash $0 user@host"

# Verify local artifacts exist
[[ -f "${LOCAL_DIR}/${TFLITE}" ]] || die "Missing ${LOCAL_DIR}/${TFLITE} — run training notebook first"
[[ -f "${LOCAL_DIR}/${META}" ]]   || die "Missing ${LOCAL_DIR}/${META}   — run training notebook first"

info "Uploading model artifacts to ${TARGET}:${REMOTE_DIR}/"
scp "${LOCAL_DIR}/${TFLITE}" "${LOCAL_DIR}/${META}" "${TARGET}:${REMOTE_DIR}/"
ok "Artifacts uploaded"

info "Restarting API containers on ${TARGET}..."
ssh "${TARGET}" "cd /opt/maize-leaf-classifier && docker compose restart api-blue api-green 2>/dev/null || docker compose restart api"
ok "Containers restarted"

info "Health check..."
# Give containers a moment to come up
sleep 3
ssh "${TARGET}" "curl -sf http://localhost:8000/health | python3 -m json.tool" \
  && ok "API is healthy" \
  || { printf '\033[1;33m[WARN]\033[0m  Health check failed — check: ssh %s docker compose logs api-blue\n' "$TARGET"; }
