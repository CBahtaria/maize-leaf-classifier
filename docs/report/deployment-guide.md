# Deployment Guide

This guide covers two deployment workflows:

1. **Standard Docker Compose** — single-slot deployment for development,
   testing, and small-scale production.
2. **Blue-green deployment** — zero-downtime production deployment with
   instant rollback.

Both target a single Ubuntu 22.04 VPS.

---

## 1. Standard Deployment (Docker Compose)

### 1.1 Server requirements

- **OS:** Ubuntu 22.04 LTS (Jammy)
- **CPU:** 2 vCPU minimum
- **RAM:** 2 GB minimum (4 GB recommended)
- **Disk:** 20 GB free
- **Network:** Public IPv4 with ports 80 and 443 open

### 1.2 Install Docker + Docker Compose

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

### 1.3 Clone the repo

```bash
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
git clone https://github.com/<your-org>/maize-leaf-classifier.git
cd maize-leaf-classifier
```

### 1.4 Place model artifacts

The model files are not committed to the public repo (they are produced by
training). Copy them into `model_artifacts/`:

```bash
# from your local machine, after training in Colab and downloading:
scp model_artifacts/mobilenetv2_best.tflite \
    model_artifacts/mobilenetv2_meta.json \
    your-user@your-vps:/opt/maize-leaf-classifier/model_artifacts/
```

Required files:

- `mobilenetv2_best.tflite` — the INT8-quantised model.
- `mobilenetv2_meta.json` — version, metrics, architecture metadata.

### 1.5 Configure environment

```bash
cp .env.example .env
nano .env
```

Set:

```env
DOMAIN=your-domain.com
ALLOWED_ORIGINS=["https://your-domain.com"]
MODEL_PATH=/app/model_artifacts/mobilenetv2_best.tflite
MODEL_META_PATH=/app/model_artifacts/mobilenetv2_meta.json
LOG_LEVEL=INFO
```

### 1.6 Start the stack

```bash
docker compose -f docker/docker-compose.yml up -d
```

This brings up `nginx`, `api-blue`, `api-green`, and the one-shot `frontend`
asset builder. The `frontend` container exits after copying the compiled bundle
into the shared volume — this is normal.

### 1.7 Verify

```bash
curl http://localhost/health
```

Expected:

```json
{"status":"ok","model_loaded":true,"model_version":"1.0.0","uptime_seconds":12.4}
```

If `status` is `"degraded"`, check `MODEL_PATH` in `.env` and the mounted
`model_artifacts/` directory.

---

## 2. Blue-Green Deployment

This is the production deployment workflow. Two API containers run
simultaneously; one is active, one is the deploy target for the next release.

### 2.1 Bootstrap (first time only)

The standard deployment above already starts both `api-blue` and `api-green`:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Initial state: `api-blue` is active by convention (no `.active-slot` file →
defaults to blue).

### 2.2 Deploy a new version

```bash
bash scripts/deploy.sh <new-image-tag>
```

For example, `bash scripts/deploy.sh abc123def` to deploy the API image built
from commit `abc123def`.

The script:

1. Reads the current active slot.
2. Pulls the new image into the **inactive** slot.
3. Restarts the inactive slot with the new image.
4. Polls `/health` on the inactive slot (up to 10 retries, 5 seconds apart).
5. On healthy → rewrites `docker/conf.d/active-upstream.conf`, runs
   `nginx -s reload`, persists the new slot to `.active-slot`. Exit 0.
6. On unhealthy → leaves traffic on the old slot, exits 1.

### 2.3 Verify the new version

```bash
curl http://localhost/health
curl http://localhost/model/info
```

Confirm `model_version` matches what you expected to deploy.

### 2.4 Rollback

The previous slot is still running. To switch back instantly:

```bash
bash scripts/rollback.sh
```

This rewrites `active-upstream.conf` back to the previous slot and reloads
nginx. **No health check is run** — we trust that the slot we just rolled off
was healthy moments ago.

### 2.5 GitHub Actions setup (continuous deployment)

The CD workflow at `.github/workflows/ci-cd.yml` builds the API image, pushes
it to `ghcr.io`, and SSHs into the VPS to run `deploy.sh`.

Required GitHub repository secrets:

| Secret | Value |
|---|---|
| `VPS_HOST` | VPS hostname or IP (e.g. `203.0.113.10`) |
| `VPS_USER` | SSH username (e.g. `deploy`) |
| `VPS_SSH_KEY` | Private SSH key with access to `/opt/maize-leaf-classifier` |

To set these:

```
GitHub repo → Settings → Secrets and variables → Actions → New repository secret
```

Then push to `main` to trigger CD. The workflow:

1. Checks out the code.
2. Builds the API image.
3. Pushes to `ghcr.io/<owner>/maize-api:<sha>`.
4. SSHs into the VPS, `cd /opt/maize-leaf-classifier`, runs
   `bash scripts/deploy.sh <sha>`.

If the deploy script fails (health check timeout), the GitHub Actions run
fails. The previous slot keeps serving traffic — production is never affected.

---

## 3. HTTPS with Certbot

The default deployment serves HTTP on port 80. For production you must enable
HTTPS.

### 3.1 Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 3.2 Obtain a certificate

Stop the Docker nginx first (Certbot needs port 80):

```bash
docker compose -f docker/docker-compose.yml stop nginx
sudo certbot certonly --standalone -d your-domain.com
```

Or use the nginx plugin if you run a host-level nginx as a TLS terminator in
front of the Docker stack:

```bash
sudo certbot --nginx -d your-domain.com
```

### 3.3 Wire the cert into the Docker nginx

Mount `/etc/letsencrypt/live/your-domain.com/` into the nginx container
(extend `docker/docker-compose.yml`), and update `docker/nginx.conf` to add a
`server` block listening on 443 with `ssl_certificate` and
`ssl_certificate_key` pointing at the mounted paths.

### 3.4 Update CORS

In `.env`, set:

```env
ALLOWED_ORIGINS=["https://your-domain.com"]
```

Then `docker compose -f docker/docker-compose.yml up -d` to apply.

### 3.5 Auto-renewal

Certbot installs a systemd timer for auto-renewal. Verify with:

```bash
sudo systemctl status certbot.timer
```

Renewal triggers a `--deploy-hook` that should reload the Docker nginx:

```bash
sudo tee /etc/letsencrypt/renewal-hooks/deploy/maize-reload.sh > /dev/null <<'EOF'
#!/bin/bash
docker compose -f /opt/maize-leaf-classifier/docker/docker-compose.yml \
  exec -T nginx nginx -s reload
EOF
sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/maize-reload.sh
```

---

## 4. Operational notes

- **Logs:** `docker compose -f docker/docker-compose.yml logs -f --tail=200`.
- **Restart a single slot:** `docker compose -f docker/docker-compose.yml restart api-blue`.
- **Update model:** copy new `.tflite` into `model_artifacts/`, run
  `docker compose ... restart api-blue api-green`. Bind mount is read-only, so
  changes on the host take effect on container restart.
- **Resource ceiling:** TFLite uses ~250 MB RAM at idle, ~400 MB under load.
  Two slots fit in 2 GB RAM with nginx, frontend volume copy, and headroom.
