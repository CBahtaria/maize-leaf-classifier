#!/usr/bin/env bash
# wire-frontend.sh — Point the Vercel frontend at the live Render backend.
#
# Usage:
#   bash scripts/wire-frontend.sh https://maizescan-api.onrender.com
#
# What this does:
#   1. Sets VITE_API_URL in Vercel (production environment)
#   2. Triggers a production redeploy
#   3. Verifies the new deployment is reachable
set -euo pipefail

RENDER_URL="${1:-}"

die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }

# ── Preflight checks ────────────────────────────────────────────────────────
[[ -n "$RENDER_URL" ]] || die "Usage: bash $0 https://maizescan-api.onrender.com"
# Strip trailing slash
RENDER_URL="${RENDER_URL%/}"
command -v vercel >/dev/null || die "'vercel' CLI not installed — run: npm i -g vercel"

# Basic URL sanity check
[[ "$RENDER_URL" =~ ^https?:// ]] || die "URL must start with https://"

info "Render backend URL: ${RENDER_URL}"

# ── Set VITE_API_URL in Vercel ───────────────────────────────────────────────
info "Setting VITE_API_URL in Vercel (production)..."

# Remove existing value if present (ignore errors)
echo "" | vercel env rm VITE_API_URL production --yes 2>/dev/null || true

# Add new value
echo "$RENDER_URL" | vercel env add VITE_API_URL production

ok "VITE_API_URL set to: ${RENDER_URL}"

# ── Redeploy frontend ────────────────────────────────────────────────────────
info "Triggering production redeploy..."
vercel deploy --prod --yes

ok "Frontend redeployed to https://maizescan.vercel.app"

# ── Smoke test ───────────────────────────────────────────────────────────────
info "Running smoke test against backend..."
sleep 3  # give Render a moment if it spun down

HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${RENDER_URL}/health" 2>/dev/null || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    ok "Backend /health → 200"
    curl -s "${RENDER_URL}/health" | python3 -m json.tool 2>/dev/null || true
elif [[ "$HTTP_CODE" == "000" ]]; then
    printf '\033[1;33m[WARN]\033[0m  Backend unreachable — Render free tier may be sleeping (first request takes ~30s)\n'
    printf '         Run manually: curl %s/health\n' "$RENDER_URL"
else
    printf '\033[1;33m[WARN]\033[0m  Backend returned HTTP %s — check Render logs\n' "$HTTP_CODE"
fi

printf '\n'
printf '\033[1;32m╔══════════════════════════════════════════════════════════════╗\033[0m\n'
printf '\033[1;32m║  All done! Live URLs                                         ║\033[0m\n'
printf '\033[1;32m╚══════════════════════════════════════════════════════════════╝\033[0m\n'
printf '\n'
printf '  Frontend : https://maizescan.vercel.app\n'
printf '  Backend  : %s\n' "$RENDER_URL"
printf '  Health   : %s/health\n' "$RENDER_URL"
printf '  API docs : %s/docs  (only in DEBUG=true)\n' "$RENDER_URL"
