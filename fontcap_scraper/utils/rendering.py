import logging
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import io

logger = logging.getLogger(__name__)

def render_glyphs(
        font_bytes: bytes, 
        charset: list[str], 
        img_size: int, 
        font_size: int) -> dict[str, Image.Image] | None:
    """Render the pulled files"""
    try:
        font = ImageFont.truetype(io.BytesIO(font_bytes), font_size)
    except Exception as e:
        logger.warning(f"Could not load font: {e}")
        return None

    glyphs = {}
    for char in charset:
        try:
            img = Image.new("L", (img_size, img_size), color=255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), char, font=font)
            if bbox is None:
                continue
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                ((img_size - w) / 2 - bbox[0], (img_size - h) / 2 - bbox[1]),
                char, font=font, fill=0
            )
            glyphs[char] = img
        # Sometimes there are random exceptions...
        except Exception as e:
            logger.warning(f"Rendering failed for char {repr(char)}: {e}")
            return None
    return glyphs if glyphs else None

def save_glyphs(font_name: str, glyphs: dict[str, Image.Image], output_dir: Path):
    font_dir = output_dir / font_name
    font_dir.mkdir(parents=True, exist_ok=True)

    for char, img in glyphs.items():
        img_path = font_dir / f"{ord(char)}.png" # Note the ord(char)
        img.save(img_path, format="PNG")
