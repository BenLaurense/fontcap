import streamlit as st

st.set_page_config(layout="wide")

import streamlit.components.v1 as components
from pathlib import Path
from PIL import Image
from random import shuffle
import io
from base64 import b64encode

"""
Simple streamlit UI to view the fonts
"""

# TODO make these configurable
DATA_ROOT = Path("data/fonts")
INDEX_FILE = DATA_ROOT / "index.json"

def load_fonts_metadata(font_root: Path, limit: int = 100) -> list:
    font_dirs = [d for d in font_root.iterdir() if d.is_dir()]
    shuffle(font_dirs)  # random order
    selected = font_dirs[:limit]

    font_data = []
    for font_dir in selected:
        font_name = font_dir.name
        glyph_files = list(font_dir.glob("*.png"))
        if not glyph_files:
            continue
        font_data.append({
            "name": font_name,
            "path": font_dir,
            "glyphs": glyph_files
        })
    return font_data

max_fonts = st.sidebar.number_input("Number of fonts to display", min_value=1, max_value=500, value=100)
fonts = load_fonts_metadata(DATA_ROOT, limit=max_fonts)

st.markdown("""
    <style>
    img {
        border-radius: 0 !important;
        image-rendering: pixelated;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def render_glyph_row(images: list[Image.Image], label: str, top_margin: int = 50):
    st.markdown(f"**{label}**", unsafe_allow_html=True)
    imgs_html = ""
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = b64encode(buf.getvalue()).decode()
        imgs_html += f'''
            <img src="data:image/png;base64,{b64}"
                 style="width:64px; height:64px;
                        image-rendering: pixelated;
                        border-radius: 0;
                        background: none;" />
        '''
    html = f"""
        <div style='
            display: flex;
            flex-wrap: wrap;
            gap: 0px;
            margin-top: {top_margin}px;
            margin-bottom: 1rem;
        '>
            {imgs_html}
        </div>
    """
    components.html(html, height=200 + top_margin, scrolling=False)

for font in fonts:
    st.markdown(f"### {font['name']}")

    uppercase = [Image.open(font["path"] / f"{ord(c)}.png") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (font["path"] / f"{ord(c)}.png").exists()]
    lowercase = [Image.open(font["path"] / f"{ord(c)}.png") for c in "abcdefghijklmnopqrstuvwxyz" if (font["path"] / f"{ord(c)}.png").exists()]

    render_glyph_row(uppercase, "Uppercase") # type: ignore
    render_glyph_row(lowercase, "Lowercase") # type: ignore
    st.markdown("---")
