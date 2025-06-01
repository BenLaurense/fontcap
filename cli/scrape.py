import logging
import click
from fontcap_scraper.main import run_scraper
from fontcap_scraper.config import FontcapConfig

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='[%(asctime)s] %(levelname)s: %(message)s',
)

@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to YAML config file')
def run(config, verbose):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = FontcapConfig.from_yaml(config)
    run_scraper(cfg)
    return

if __name__ == '__main__':
    run()
