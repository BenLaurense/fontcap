import io
import requests
from pathlib import Path
from config import FontcapConfig
from sources.google_fonts import GoogleFontsSource
from utils.deduplication import load_known_hashes
from utils.rendering import render_glyphs, save_glyphs


def run_scraper(cfg: FontcapConfig):
    """Scrape fonts according to the supplied configuration"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load known hashes
    known_hashes = load_known_hashes(cfg.index_file)

    for source in cfg.font_sources:
        if source["name"] == "google_fonts":
            font_source = GoogleFontsSource(source["url"], source["api_key"])
        else:
            raise NotImplementedError(f"Source '{source['name']}' not implemented")

        # Get list of all fonts available
        font_metadata_list = []
        font_metadata_list = font_source.fetch_font_list()

        # Download each font file if it doesn't exist
        for metadata in font_metadata_list[:2]:
            font_bytes = download_font_file(metadata.url)
            if not font_bytes:
                continue

            hsh = font_hash(font_bytes)
            if hsh in known_hashes:
                continue

            glyphs = render_glyphs(font_bytes, cfg.charset, cfg.img_size, cfg.font_size)
            if glyphs is None or len(glyphs) < cfg.min_chars_required:
                continue

            save_glyphs(metadata.name, glyphs, cfg.output_dir)
            known_hashes.add(hsh)

    # Update and store hashes
    update_known_hashes(cfg.index_file, known_hashes)
    return


def download_font_file(url: str) -> bytes:
    """Attemps to download the font .tff file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Failed to download font from {url}: {e}")
        return b""

if __name__ == '__main__':
    here = Path(__file__).parent
    config_path = here / "basic_config.yaml"
    cfg = FontcapConfig.from_yaml(config_path)
    print(cfg)
