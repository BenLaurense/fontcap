import streamlit as st
from pathlib import Path
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, default="data/fonts")
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
INDEX_FILE = DATA_ROOT / "index.json"

st.set_page_config(layout="wide")

def load_fonts(font_root: Path) -> list:
    font_data = []
    for font_dir in font_root.iterdir():
        if font_dir.is_dir():
            font_name = font_dir.name
            glyph_files = list(font_dir.glob("*.png"))
            if not glyph_files:
                continue
            font_data.append({
                "name": font_name,
                "path": font_dir,
                "glyphs": glyph_files,
            })
    return font_data

fonts = load_fonts(DATA_ROOT)

for font in fonts:
    st.markdown(f"### {font['name']}")

    uppercase = [Image.open(font["path"] / f"{ord(c)}.png") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (font["path"] / f"{ord(c)}.png").exists()]
    lowercase = [Image.open(font["path"] / f"{ord(c)}.png") for c in "abcdefghijklmnopqrstuvwxyz" if (font["path"] / f"{ord(c)}.png").exists()]

    st.image(uppercase, width=32)
    st.image(lowercase, width=32)
    st.markdown("---")
