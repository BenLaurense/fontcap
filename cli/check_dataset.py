import click
from pathlib import Path
from PIL import Image
import numpy as np

"""
CLI tool to interrogate scraped fonts
"""

@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def check_dataset(dataset_dir: Path):
    """Check dataset structure and compute per-font metrics (density, sparseness)."""
    click.echo(f"Checking dataset in: {dataset_dir}")
    font_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]

    total_fonts = len(font_dirs)
    metrics = []

    for font_dir in font_dirs:
        png_files = list(font_dir.glob("*.png"))
        if not png_files:
            click.echo(f"[EMPTY] {font_dir.name}")
            continue

        density_values = []
        sparseness_values = []

        for img_path in png_files:
            try:
                with Image.open(img_path) as img:
                    if img.mode != "L":
                        raise ValueError("Image is not in grayscale (mode 'L')")

                    density = compute_density(img)
                    density_values.append(density)

                    sparseness = compute_sparseness(img)
                    sparseness_values.append(sparseness)

            except Exception as e:
                click.echo(f"[CORRUPT] {font_dir.name} ({img_path.name}): {e}")
                break
        else:
            avg_density = np.mean(density_values)
            avg_sparseness = np.mean(sparseness_values)
            metrics.append((font_dir.name, avg_density, avg_sparseness))

            density_color = color_metric(avg_density, 0.05, 0.25) # type: ignore
            sparse_color = color_metric(avg_sparseness, 1.5, 5.0) # type: ignore
            density_str = click.style(f"{avg_density:.3f}", fg=density_color)
            sparse_str = click.style(f"{avg_sparseness:.2f}", fg=sparse_color)
            click.echo(f"[SUCCESS] {font_dir.name:<30} | density={density_str} | sparseness={sparse_str}")

    click.echo("\nSummary:")
    click.echo(f"Fonts checked: {total_fonts}")
    click.echo(f"Fonts processed successfully: {len(metrics)}\n")

def color_metric(value: float, low: float, high: float) -> str:
    """Colorize a value from red (low) to green (high) using ANSI colors."""
    if value < low:
        color = "red"
    elif value < (low + high) / 2:
        color = "yellow"
    else:
        color = "green"
    return color

def compute_density(image: Image.Image) -> float:
    """Pixel density"""
    arr = np.array(image)
    binary = arr < 128  # Treat black-ish pixels as "on"
    return binary.mean()

def compute_sparseness(image: Image.Image) -> float:
    """Sparseness: the number of black pixels near each black pixels"""
    arr = np.array(image)
    binary = arr < 128
    black_pixels = np.argwhere(binary)
    if len(black_pixels) == 0:
        return 0.0

    padded = np.pad(binary, 1, mode='constant')
    sparseness = 0
    for y, x in black_pixels:
        neighborhood = padded[y:y+3, x:x+3]
        neighbors = np.sum(neighborhood) - 1
        sparseness += neighbors
    return sparseness / len(black_pixels) # type: ignore

if __name__ == "__main__":
    check_dataset()
