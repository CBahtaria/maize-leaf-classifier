"""Image file validation middleware.

Security hardening for file upload endpoint:
1. Content-Type whitelist (not just extension — prevents MIME sniffing attacks)
2. Hard size cap before full read (prevents memory exhaustion)
3. PIL verify() — detects corrupted headers, truncated files, decompression bombs
4. PIL load() — fully decodes the image (verify() exhausts its internal buffer)
5. RGB conversion — strips EXIF metadata (GPS location from farmer field photos)
6. Returns clean sanitised bytes — original bytes never used after validation

Privacy note: Field photos from SSA farmers often contain GPS EXIF metadata. Converting
to RGB via PIL strips all metadata before inference, protecting farmer privacy.
"""

import io
import logging

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)

MAX_IMAGE_DIMENSION = 10_000  # pixels — reject absurdly large dimensions


async def validate_image_file(
    file: UploadFile,
    max_bytes: int = 10 * 1024 * 1024,
) -> bytes:
    """Validate an uploaded image file and return sanitised JPEG bytes.

    Raises:
        HTTPException 415: Unsupported media type (not an allowed image format)
        HTTPException 413: File too large (exceeds max_bytes)
        HTTPException 400: Invalid or corrupt image data

    Returns:
        Sanitised JPEG bytes (EXIF stripped, RGB normalised).
    """
    # 1. Check Content-Type header
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected upload: unsupported content-type '%s'", content_type)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # 2. Read with hard size cap (prevents reading arbitrarily large files into memory)
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        logger.warning(
            "Rejected upload: file size %d bytes exceeds limit %d", len(content), max_bytes
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size: {max_bytes // (1024 * 1024)} MB",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # 3. PIL verify() — checks for corrupt headers, truncated files, decompression bombs
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning("Rejected upload: PIL verify failed — %s", exc)
        raise HTTPException(status_code=400, detail="Invalid or corrupt image data") from exc

    # 4. Re-open and load() — verify() exhausts internal buffer, must reopen
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
    except Exception as exc:
        logger.warning("Rejected upload: PIL load failed — %s", exc)
        raise HTTPException(status_code=400, detail="Could not decode image") from exc

    # Sanity check on dimensions (reject decompression-bomb-sized images)
    if max(img.size) > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large (max {MAX_IMAGE_DIMENSION}px per side)",
        )

    # 5. Convert to RGB — strips ALL EXIF metadata (including GPS location)
    #    Returns clean image with no provenance information from the original file
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    sanitised = buf.getvalue()

    logger.debug(
        "Validated image: %dx%d → %d bytes sanitised JPEG", img.width, img.height, len(sanitised)
    )
    return sanitised
