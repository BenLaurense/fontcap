import io
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

def render_glyphs(
        font_bytes: bytes, 
        charset: list[str], 
        img_size: int, 
        font_size: int) -> dict[str, Image.Image] | None:
    """Attempts to render the font as images"""
    try:
        font = ImageFont.truetype(io.BytesIO(font_bytes), font_size)
    except Exception as e:
        print(f"Could not load font: {e}")
        return None

    glyphs = {}
    for char in charset:
        img = Image.new("L", (img_size, img_size), color=255)
        draw = ImageDraw.Draw(img)
        # w, h = draw.textsi(char, font=font)
        w = draw.textlength(char, font=font)
        h = img_size
        draw.text(((img_size - w) / 2, (img_size - h) / 2), char, font=font, fill=0)
        glyphs[char] = img

    return glyphs

def save_glyphs(
        font_name: str, 
        glyphs: dict[str, Image.Image], 
        output_dir: Path) -> None:
    font_dir = output_dir / font_name
    font_dir.mkdir(parents=True, exist_ok=True)

    for char, img in glyphs.items():
        # Use ord(char) as filename to ensure filesystem compatibility
        img_path = font_dir / f"{ord(char)}.png"
        img.save(img_path, format="PNG")
    return
