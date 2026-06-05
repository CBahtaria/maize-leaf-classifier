#!/usr/bin/env bash
# Blue-green deployment switcher for maize-leaf-classifier.
# Usage: ./scripts/deploy.sh [new-image-tag]
# Exit codes: 0 = success, 1 = health check failure (traffic stays on current slot)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLOT_FILE="${PROJECT_ROOT}/.active-slot"
CONF_FILE="${PROJECT_ROOT}/docker/conf.d/active-upstream.conf"

NEW_TAG="${1:-latest}"
MAX_RETRIES=10
RETRY_INTERVAL=5

# Determine current (active) and target (inactive) slots
CURRENT_SLOT="blue"
if [[ -f "${SLOT_FILE}" ]]; then
    CURRENT_SLOT="$(cat "${SLOT_FILE}" | tr -d '[:space:]')"
fi
if [[ "${CURRENT_SLOT}" == "blue" ]]; then
    INACTIVE_SLOT="green"
else
    INACTIVE_SLOT="blue"
fi

echo "[deploy] Current: ${CURRENT_SLOT} → Target: ${INACTIVE_SLOT} (tag: ${NEW_TAG})"
cd "${PROJECT_ROOT}"
export API_TAG="${NEW_TAG}"

# Pull new image (|| true: don't fail for local-only builds)
docker compose -f docker/docker-compose.yml pull "api-${INACTIVE_SLOT}" 2>&1 || true

# Restart inactive slot with the new image
docker compose -f docker/docker-compose.yml up -d --no-deps --force-recreate "api-${INACTIVE_SLOT}"

# Poll health of the inactive slot directly (not through nginx)
HEALTHY=false
for i in $(seq 1 ${MAX_RETRIES}); do
    echo "[deploy] Health check ${i}/${MAX_RETRIES}..."
    STATUS=$(docker compose -f docker/docker-compose.yml \
        exec -T "api-${INACTIVE_SLOT}" \
        python -c "
import urllib.request, json, sys
try:
    r = urllib.request.urlopen('http://localhost:8000/health', timeout=8)
    d = json.loads(r.read())
    print(d.get('status', 'unknown'))
except Exception:
    sys.exit(1)
" 2>/dev/null) || STATUS="error"

    if [[ "${STATUS}" == "ok" ]]; then
        HEALTHY=true
        echo "[deploy] api-${INACTIVE_SLOT} is healthy!"
        break
    fi
    echo "[deploy] Not ready yet (${STATUS}). Waiting ${RETRY_INTERVAL}s..."
    sleep "${RETRY_INTERVAL}"
done

if [[ "${HEALTHY}" != "true" ]]; then
    echo "[deploy] FAILED: api-${INACTIVE_SLOT} did not become healthy after ${MAX_RETRIES} attempts."
    echo "[deploy] Traffic remains on api-${CURRENT_SLOT}. No changes made to nginx."
    exit 1
fi

# Switch nginx upstream to the new slot
mkdir -p "$(dirname "${CONF_FILE}")"
cat > "${CONF_FILE}" <<EOF
# Managed by deploy.sh — last updated: $(date -u +"%Y-%m-%dT%H:%M:%SZ") — slot: ${INACTIVE_SLOT}
upstream active_api {
    server api-${INACTIVE_SLOT}:8000;
}
EOF

docker compose -f docker/docker-compose.yml exec -T nginx nginx -s reload
echo "${INACTIVE_SLOT}" > "${SLOT_FILE}"

echo "[deploy] Done. Active slot: ${INACTIVE_SLOT} (tag: ${NEW_TAG})."
echo "[deploy] Previous slot (api-${CURRENT_SLOT}) still running — run rollback.sh to revert."
