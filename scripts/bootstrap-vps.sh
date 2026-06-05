#!/usr/bin/env bash
# bootstrap-vps.sh — one-command Ubuntu 22.04 VPS provisioning
# Usage: sudo bash scripts/bootstrap-vps.sh
set -euo pipefail

REPO_URL="https://github.com/charleskris9/maize-leaf-classifier.git"
DEPLOY_DIR="/opt/maize-leaf-classifier"
DEPLOY_USER="deploy"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash $0"
[[ $(lsb_release -rs 2>/dev/null) == "22.04" ]] || warn "Tested on Ubuntu 22.04 — current OS may differ"

# ── 1. Docker Engine + Compose plugin ────────────────────────────────────────
info "Installing Docker Engine..."
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker
ok "Docker $(docker --version | awk '{print $3}' | tr -d ,) installed"

# ── 2. deploy user ────────────────────────────────────────────────────────────
info "Creating '${DEPLOY_USER}' user..."
if id "${DEPLOY_USER}" &>/dev/null; then
  warn "User '${DEPLOY_USER}' already exists — skipping creation"
else
  adduser --disabled-password --gecos "" "${DEPLOY_USER}"
fi
usermod -aG docker "${DEPLOY_USER}"

# Copy root's authorized_keys so the deploy user can SSH in
ROOT_KEYS="/root/.ssh/authorized_keys"
DEPLOY_SSH="/home/${DEPLOY_USER}/.ssh"
if [[ -f "$ROOT_KEYS" ]]; then
  mkdir -p "$DEPLOY_SSH"
  cp "$ROOT_KEYS" "$DEPLOY_SSH/authorized_keys"
  chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "$DEPLOY_SSH"
  chmod 700 "$DEPLOY_SSH"
  chmod 600 "$DEPLOY_SSH/authorized_keys"
  ok "SSH authorized_keys copied to ${DEPLOY_USER}"
else
  warn "No root authorized_keys found — add SSH keys to ${DEPLOY_SSH}/authorized_keys manually"
fi

# ── 3. Clone repository ───────────────────────────────────────────────────────
info "Cloning repository to ${DEPLOY_DIR}..."
if [[ -d "${DEPLOY_DIR}/.git" ]]; then
  warn "Repository already exists at ${DEPLOY_DIR} — pulling latest"
  git -C "${DEPLOY_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${DEPLOY_DIR}"
fi
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "${DEPLOY_DIR}"
ok "Repository ready at ${DEPLOY_DIR}"

# ── 4. Create model_artifacts dir + .env ─────────────────────────────────────
info "Setting up runtime directories..."
cd "${DEPLOY_DIR}"
mkdir -p model_artifacts
if [[ ! -f .env ]]; then
  cp .env.example .env
  ok ".env created from .env.example — edit before first deploy"
else
  warn ".env already exists — not overwritten"
fi
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" model_artifacts .env 2>/dev/null || true

# ── 5. Post-bootstrap checklist ──────────────────────────────────────────────
cat <<'CHECKLIST'

╔══════════════════════════════════════════════════════════════════════════════╗
║               POST-BOOTSTRAP CHECKLIST                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Upload model artifacts:                                                  ║
║       bash scripts/upload-model.sh deploy@<your-server-ip>                  ║
║                                                                              ║
║  2. Edit environment config:                                                 ║
║       nano /opt/maize-leaf-classifier/.env                                   ║
║     Set: ALLOWED_ORIGINS, MODEL_PATH, API_KEY (if used)                      ║
║                                                                              ║
║  3. First deploy:                                                            ║
║       su - deploy                                                            ║
║       cd /opt/maize-leaf-classifier                                          ║
║       docker compose up -d                                                   ║
║                                                                              ║
║  4. Add GitHub Secrets for CI/CD:                                            ║
║       VPS_HOST   — server IP or hostname                                     ║
║       VPS_USER   — deploy                                                    ║
║       VPS_KEY    — private SSH key (matching authorized_keys above)          ║
║                                                                              ║
║  5. Verify services:                                                         ║
║       docker compose ps                                                      ║
║       curl http://localhost:8000/health                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CHECKLIST
