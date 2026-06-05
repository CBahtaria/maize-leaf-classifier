#!/usr/bin/env bash
# lighthouse.sh — run Lighthouse audit and open HTML report
# Usage: bash scripts/lighthouse.sh https://yourdomain.com
set -euo pipefail

URL="${1:-}"
REPORT="./lighthouse-report.html"

die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }

[[ -n "$URL" ]] || die "Usage: bash $0 https://yourdomain.com"

# Dependency checks
command -v node  &>/dev/null || die "Node.js not found — install from https://nodejs.org"
command -v npx   &>/dev/null || die "npx not found — update Node.js"

info "Running Lighthouse audit on ${URL}..."
npx --yes lighthouse "${URL}" \
  --output html \
  --output-path "${REPORT}" \
  --chrome-flags="--headless --no-sandbox" \
  --quiet

ok "Report saved to ${REPORT}"

# Extract summary scores from JSON sidecar if available (lighthouse ≥ 10 skips JSON by default)
JSON_REPORT="${REPORT%.html}.json"
if [[ -f "$JSON_REPORT" ]]; then
  python3 - "$JSON_REPORT" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
cats = data.get("categories", {})
labels = {"performance": "Performance", "accessibility": "Accessibility",
          "best-practices": "Best Practices", "seo": "SEO", "pwa": "PWA"}
print("\n── Lighthouse Scores ──────────────────")
for k, label in labels.items():
    score = cats.get(k, {}).get("score")
    if score is not None:
        pct = int(score * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        color = "\033[1;32m" if pct >= 90 else "\033[1;33m" if pct >= 50 else "\033[1;31m"
        print(f"  {label:<16} {color}{pct:>3}%\033[0m  {bar}")
print()
PY
fi

# Open report in browser if possible
if command -v xdg-open &>/dev/null; then
  xdg-open "${REPORT}" 2>/dev/null &
elif command -v open &>/dev/null; then
  open "${REPORT}"
else
  info "Open manually: ${REPORT}"
fi
