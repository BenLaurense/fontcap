import logging
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
from fontcap_model.dataset import get_dataloaders
from models import CNN_Autoencoder
from utils import plot_losses, display_reconstructions

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
    
def train_autoencoder(
    data_root: str | Path,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_interval: int = 5,
    plot_interval: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")
    model = CNN_Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader, val_loader = get_dataloaders(data_root, batch_size=batch_size)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for lower, upper in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            lower, upper = lower.to(device), upper.to(device)

            optimizer.zero_grad()
            outputs = model(lower)
            loss = criterion(outputs, upper)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * lower.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset) # type: ignore
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for lower, upper in val_loader:
                lower, upper = lower.to(device), upper.to(device)
                outputs = model(lower)
                loss = criterion(outputs, upper)
                epoch_val_loss += loss.item() * lower.size(0)

        val_loss = epoch_val_loss / len(val_loader.dataset) # type: ignore
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {learning_rate}")

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"autoencoder_epoch{epoch}.pt")

        if epoch % plot_interval == 0:
            plot_losses(train_losses, val_losses, checkpoint_dir / "loss_curve.png")
            display_reconstructions(model, val_loader, device, checkpoint_dir / f"recon_epoch{epoch}.png")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG, WARNING, etc.
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )
    train_autoencoder(data_root="data/fonts", num_epochs=20)
