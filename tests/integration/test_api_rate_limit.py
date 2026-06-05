"""Rate limiting tests — 21st request within a minute must be 429."""


def test_rate_limit_enforced(test_client, sample_healthy_image):
    """First 20 requests succeed; 21st must be 429 with Retry-After header.

    Note: slowapi uses get_remote_address which resolves to '127.0.0.1' (or 'testclient')
    for FastAPI TestClient. The rate limit is scoped per IP, so all TestClient requests
    share the same bucket and the 21st should trigger 429.
    """
    for i in range(20):
        r = test_client.post(
            "/predict",
            files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
        )
        assert r.status_code == 200, f"Request {i + 1} failed with {r.status_code}: {r.text}"

    r = test_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
    )
    assert r.status_code == 429, f"Expected 429 on 21st request, got {r.status_code}"
    assert "Retry-After" in r.headers, "Missing Retry-After header on 429"
