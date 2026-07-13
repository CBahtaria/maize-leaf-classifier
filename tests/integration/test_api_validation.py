"""Security validation tests — file type, size, corruption."""


def test_rejects_non_image_mime(test_client):
    """415 for non-image MIME type."""
    r = test_client.post(
        "/predict",
        files={"file": ("evil.exe", b"MZ\x90\x00\x03\x00", "application/octet-stream")},
    )
    assert r.status_code == 415, f"Expected 415, got {r.status_code}"


def test_rejects_oversized_file(test_client):
    """413 for file > MAX_FILE_SIZE_MB (10 MB)."""
    # 11 MB of fake JPEG data
    big = b"\xff\xd8\xff\xe0" + b"0" * (11 * 1024 * 1024)
    r = test_client.post(
        "/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413, f"Expected 413, got {r.status_code}"


def test_rejects_corrupt_image(test_client):
    """400 for a file with correct MIME but corrupt image data."""
    r = test_client.post(
        "/predict",
        files={"file": ("bad.jpg", b"\xff\xd8\xff\x00corrupt_data_here", "image/jpeg")},
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"


def test_accepts_jpeg(test_client, sample_healthy_image):
    r = test_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_healthy_image, "image/jpeg")},
    )
    assert r.status_code == 200


def test_accepts_png(test_client, sample_healthy_image):
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(sample_healthy_image))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = test_client.post(
        "/predict",
        files={"file": ("leaf.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
