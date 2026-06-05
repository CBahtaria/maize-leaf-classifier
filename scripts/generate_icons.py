from PIL import Image, ImageDraw
import os

os.makedirs('frontend/public/icons', exist_ok=True)

def generate_icon(size, path):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark green circle background
    draw.ellipse([0, 0, size - 1, size - 1], fill='#2d6a4f')
    # White leaf oval
    p = size // 8
    draw.ellipse([p, p // 2, size - p, size - p * 2], fill='white')
    img.save(path, 'PNG')
    print(f'Generated {path}')

generate_icon(192, 'frontend/public/icons/icon-192.png')
generate_icon(512, 'frontend/public/icons/icon-512.png')
