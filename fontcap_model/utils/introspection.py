import torch
from fontcap_model.dataset import EnrichedFontcapDataset
from fontcap_model.models import CNNAutoencoder, UNet


def extract_latents(
        model: CNNAutoencoder | UNet,
        dataloader: EnrichedFontcapDataset,
        device,
        num=500):
    """
    Extract latent representation from the bottleneck layers of the CNN or U-Net model
    """
    model.eval()
    model.to(device)

    all_latents = []
    all_fonts = []
    all_chars = []
    all_b64s = []

    with torch.no_grad():
        limited = iter(dataloader)
        for i in range(num):
            lower_batch, _, font_names, chars, b64s = next(limited)
            lower_batch = lower_batch.to(device)

            # Forward pass through encoder
            encoded = model.encoder(lower_batch)              # [B, C, H, W]
            latent_vecs = encoded.view(encoded.size(0), -1)   # Flatten to [B, N]

            all_latents.append(latent_vecs)
            all_fonts.extend(font_names)
            all_chars.extend(chars)
            all_b64s.extend(b64s)

    # Stack latents
    latents = torch.cat(all_latents, dim=0).numpy()

    return latents, all_fonts, all_chars, all_b64s
