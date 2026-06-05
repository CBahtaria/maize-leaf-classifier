"""Shared pytest fixtures."""
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg_bytes(r: int, g: int, b: int) -> bytes:
    arr = np.full((224, 224, 3), [r, g, b], dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def sample_healthy_image() -> bytes:
    return _make_jpeg_bytes(34, 139, 34)


@pytest.fixture
def sample_diseased_image() -> bytes:
    return _make_jpeg_bytes(139, 69, 19)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the in-memory rate limiter storage before every test.

    The module-level limiter singleton persists its counter across all tests
    in the same pytest session. Without this reset, tests running after the
    first ~20 /predict requests get spurious 429 responses.

    Handles both limits library versions:
      2.x — counts stored in MemoryStorage.storage (OrderedDict)
      3.x — counts stored in MemoryStorage._events (dict of deques)
    """
    from api.middleware.rate_limit import limiter
    st = limiter._storage
    # Clear whichever dict attributes exist (varies by limits version)
    for attr in ("storage", "_events", "events", "expirations"):
        obj = getattr(st, attr, None)
        if isinstance(obj, dict):
            obj.clear()
    yield


@pytest.fixture
def test_client(monkeypatch):
    import api.dependencies as deps

    mock_model = MagicMock()
    # predict_image() calls model.predict_raw(img_array) which returns a float
    # confidence = 0.08 → label = "Healthy" (< 0.5 threshold)
    mock_model.predict_raw.return_value = 0.08

    monkeypatch.setattr(deps, "_model", mock_model)
    monkeypatch.setattr(
        deps,
        "_model_meta",
        {"version": "test-v1", "arch_name": "mobilenetv2", "metrics": {}},
    )

    from api.main import create_app
    app = create_app()
    return TestClient(app)
