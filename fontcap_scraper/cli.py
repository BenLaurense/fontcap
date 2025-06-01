import click
from fontcap_scraper.main import run_scraper
from fontcap_scraper.config import FontcapConfig

@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Path to YAML config file')
def main(config):
    cfg = FontcapConfig.from_yaml(config)
    run_scraper(cfg)
    return
