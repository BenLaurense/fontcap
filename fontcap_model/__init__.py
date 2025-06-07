"""
Top-level package for fontcap_model
"""

from .dataset import FontcapDataset, EnrichedFontcapDataset, get_dataloaders
from .train_cnn_autoencoder import train_cnn_autoencoder
from .train_unet import train_unet
from .utils import plot_losses, display_reconstructions, extract_latents
from .models import CNNAutoencoder, UNet

__all__ = [
    "FontcapDataset", "EnrichedFontcapDataset", "get_dataloaders",
    "train_cnn_autoencoder",
    "train_unet",
    "plot_losses", "display_reconstructions", "extract_latents",
    "CNNAutoencoder", "UNet"
]
