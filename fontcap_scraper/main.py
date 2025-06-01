import logging
from tqdm import tqdm
import requests
from fontcap_scraper.config import FontcapConfig
from fontcap_scraper.utils import render_glyphs, save_glyphs, font_hash, load_known_hashes, update_known_hashes
from fontcap_scraper.sources import GoogleFontsSource

logger = logging.getLogger(__name__)

def run_scraper(cfg: FontcapConfig):
    """Scrape fonts according to the supplied configuration"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.index_file.exists():
        logger.warning(f"Index file does not exist at the specified location {cfg.index_file}. A new one will be created")
    
    # Load known hashes
    known_hashes = load_known_hashes(cfg.index_file)

    for source in cfg.font_sources:
        logger.info(f"Scraping source {source}...")
        if source["name"] == "google_fonts":
            font_source = GoogleFontsSource(source["url"], source["api_key"])
        else:
            raise NotImplementedError(f"Source '{source['name']}' not implemented")

        # Get list of all fonts available
        font_metadata_list = []
        font_metadata_list = font_source.fetch_font_list()
        logger.info(f"Found {len(font_metadata_list)} fonts")

        # Download each font file if it doesn't exist
        for metadata in tqdm(font_metadata_list[:2]):
            font_bytes = download_font_file(metadata.url)
            if not font_bytes:
                logger.info(f"Unsuccessful scrape of {metadata.name}: download failed")
                continue

            hsh = font_hash(font_bytes)
            if hsh in known_hashes:
                logger.info(f"{metadata.name} already known")
                continue

            glyphs = render_glyphs(font_bytes, cfg.charset, cfg.img_size, cfg.font_size)
            if glyphs is None or len(glyphs) < cfg.min_chars_required:
                logger.info(f"Unsuccessful scrape of {metadata.name}: not enough glyphs")
                continue

            save_glyphs(metadata.name, glyphs, cfg.output_dir)
            known_hashes.add(hsh)
            logger.info(f"Successfully scraped font {metadata.name}")
        
        logger.info(f"Scraping source {source} complete!")
        
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
    