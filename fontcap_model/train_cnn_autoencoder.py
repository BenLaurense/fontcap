import logging
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
from fontcap_model.dataset import get_dataloaders
from fontcap_model.models import CNN_Autoencoder
from fontcap_model.utils import plot_losses, display_reconstructions

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
    
def train_autoencoder(
    data_root: str | Path,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "./checkpoints",
    checkpoint_interval: int = 5, # Saves model params every x epochs
    plot_interval: int = 1, # Plots every x epochs
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    model = CNN_Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loader, test_loader = get_dataloaders(data_root, batch_size=batch_size, shuffle=True)

    train_losses, test_losses = [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        # Training loop
        for lower, upper in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            lower, upper = lower.to(device), upper.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(lower), upper)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * lower.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset) # type: ignore
        train_losses.append(train_loss)

        # Testing loss
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for lower, upper in test_loader:
                lower, upper = lower.to(device), upper.to(device)
                
                loss = loss_fn(model(lower), upper)
                epoch_test_loss += loss.item() * lower.size(0)

        test_loss = epoch_test_loss / len(test_loader.dataset) # type: ignore
        test_losses.append(test_loss)

        # Log/plot stuff
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | LR: {learning_rate}")
        if not epoch % checkpoint_interval:
            torch.save(model.state_dict(), checkpoint_dir / f"autoencoder_epoch{epoch}.pt")
        if not epoch % plot_interval:
            plot_losses(train_losses, test_losses, checkpoint_dir / "loss_curve.png")
            display_reconstructions(model, test_loader, device, checkpoint_dir / f"recon_epoch{epoch}.png")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG, WARNING, etc.
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

    train_autoencoder(data_root="data/fonts", num_epochs=10, batch_size=16)
