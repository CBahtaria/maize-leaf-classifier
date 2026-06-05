"""Integration tests for POST /predict and GET /health."""


def test_predict_returns_200(test_client, sample_healthy_image):
    r = test_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


def test_predict_response_schema(test_client, sample_healthy_image):
    r = test_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
    )
    body = r.json()
    assert body["label"] in ("Healthy", "Diseased"), f"Unexpected label: {body['label']}"
    assert 0.0 <= body["confidence"] <= 1.0, f"Confidence out of range: {body['confidence']}"
    assert body["processing_time_ms"] >= 0
    assert "model_version" in body


def test_predict_returns_model_version(test_client, sample_healthy_image):
    r = test_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
    )
    assert r.json()["model_version"] == "test-v1"


def test_health_endpoint_ok(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert isinstance(body["model_loaded"], bool)
    assert "uptime_seconds" in body
