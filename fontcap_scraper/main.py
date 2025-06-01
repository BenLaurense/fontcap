import json
from pathlib import Path
from fontcap_scraper.config import FontcapConfig

def run_scraper(cfg: FontcapConfig):
    '''
    Font scraper class
    '''
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load known hashes
    # known_hashes = load_known_hashes(cfg.index_file)

    for source in cfg.font_sources:
        if source["name"] == "google_fonts":
            font_source = None
            # font_source = GoogleFontsSource(source["url"], source.get("api_key"))
        else:
            raise NotImplementedError(f"Source '{source['name']}' not implemented")

        # Get list of all fonts available
        font_metadata_list = []
        # font_metadata_list = font_source.fetch_font_list()

        for meta in font_metadata_list:
            font_bytes = None
            # font_bytes = download_font_file(meta.url)
            if not font_bytes:
                continue

            # hsh = font_hash(font_bytes)
            # if hsh in known_hashes:
            #     continue

            # glyphs = render_glyphs(font_bytes, cfg.charset, cfg.img_size, cfg.font_size)
            # if glyphs is None or len(glyphs) < cfg.min_chars_required:
            #     continue

            # save_glyphs(meta.name, glyphs, cfg.output_dir)
            # known_hashes.add(hsh)

    # Update and store hashes
    # update_known_hashes(cfg.index_file, known_hashes)
