"""Generate PWA icons (192×192 and 512×512) as simple green-leaf PNGs."""
from pathlib import Path
from PIL import Image, ImageDraw

ICONS_DIR = Path(__file__).parent.parent / "frontend" / "public" / "icons"
ICONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_icon(size: int, dest: Path) -> None:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, size - 1, size - 1], fill="#2d6a4f")
    p = size // 8
    draw.ellipse([p, p // 2, size - p, size - p * 2], fill="white")
    img.save(dest, "PNG")
    print(f"Generated {dest} ({size}x{size})")


generate_icon(192, ICONS_DIR / "icon-192.png")
generate_icon(512, ICONS_DIR / "icon-512.png")
