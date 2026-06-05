"""Rate limiting configuration using slowapi.

Limits:
  - POST /predict: 20 requests per minute per IP
  - POST /predict/batch: 20 requests per minute per IP (shared limit with /predict)

The limiter instance is created here and imported by routes and main.py.
Exceeding the limit returns HTTP 429 with Retry-After: 60 header.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared limiter instance — imported by api/main.py and route handlers
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    headers_enabled=True,  # adds X-RateLimit-* headers to responses
)

PREDICT_LIMIT = "20/minute"
BATCH_LIMIT = "20/minute"
