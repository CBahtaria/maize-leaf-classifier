#!/usr/bin/env bash
# Immediate rollback to previous slot — no health check (assumes previous slot is healthy).
# Usage: ./scripts/rollback.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLOT_FILE="${PROJECT_ROOT}/.active-slot"
CONF_FILE="${PROJECT_ROOT}/docker/conf.d/active-upstream.conf"

if [[ ! -f "${SLOT_FILE}" ]]; then
    echo "[rollback] ERROR: .active-slot file not found. Cannot determine current slot."
    exit 1
fi

CURRENT_SLOT="$(cat "${SLOT_FILE}" | tr -d '[:space:]')"
if [[ "${CURRENT_SLOT}" == "blue" ]]; then
    PREV_SLOT="green"
else
    PREV_SLOT="blue"
fi

echo "[rollback] Rolling back: ${CURRENT_SLOT} → ${PREV_SLOT}"

cat > "${CONF_FILE}" <<EOF
# Managed by rollback.sh — last updated: $(date -u +"%Y-%m-%dT%H:%M:%SZ") — slot: ${PREV_SLOT} (ROLLBACK)
upstream active_api {
    server api-${PREV_SLOT}:8000;
}
EOF

cd "${PROJECT_ROOT}"
docker compose -f docker/docker-compose.yml exec -T nginx nginx -s reload
echo "${PREV_SLOT}" > "${SLOT_FILE}"
echo "[rollback] Done. Active slot is now: ${PREV_SLOT}"
