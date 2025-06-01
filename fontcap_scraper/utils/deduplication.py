import json
from pathlib import Path
from hashlib import sha256

def font_hash(font_bytes: bytes) -> str:
    return sha256(font_bytes).hexdigest()

def load_known_hashes(index_file: Path) -> set:
    if not index_file.exists():
        return set()
    with open(index_file, 'r') as f:
        return set(json.load(f))

def update_known_hashes(index_file: Path, hashes: set):
    with open(index_file, 'w') as f:
        json.dump(list(hashes), f)
        return
