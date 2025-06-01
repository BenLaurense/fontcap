import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class FontcapConfig:
    """Configuration class for the scraper. Usually loaded from YAML"""
    font_sources: list[dict[str, str]]
    output_dir: Path
    charset: list[str]
    img_size: int
    font_size: int
    min_chars_required: int
    index_file: Path

    def __post_init__(self):
        self.index_file = self.output_dir / self.index_file

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        data.setdefault('charset', [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)])
        data['output_dir'] = Path(data['output_dir'])
        return cls(**data)
