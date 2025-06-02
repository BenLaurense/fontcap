import logging
import json
import torch
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
from fontcap_model.dataset import get_dataloaders
from fontcap_model.models import UNet
from fontcap_model.utils import plot_losses, display_reconstructions

logger = logging.getLogger(__name__)


def train_unet(
        data_root: str | Path,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        checkpoint_dir: str | Path,
        checkpoint_interval: int = 5,
        plot_interval: int = 1,
        state_dict_name: str | None = None,
        resume_loss: bool = False
):
    """
    Training loop for the CNN_Autoencoder model.
    A new checkpoints directory should be created each run or they will
    overwrite, e.g. ./checkpoints_cnn. Sensible defaults:
    batch_size: 32
    learning_rate: 1e-3
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = UNet().to(device)
    if state_dict_name:
        model.load_state_dict(torch.load(checkpoint_dir / state_dict_name))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loader, test_loader = get_dataloaders(data_root, batch_size=batch_size, shuffle=True)

    # Load losses if they exist
    train_loss_path = checkpoint_dir / "train_losses.json"
    test_loss_path = checkpoint_dir / "test_losses.json"
    train_losses, test_losses = [], []
    if resume_loss and train_loss_path.exists() and test_loss_path.exists():
        with open(train_loss_path, "r") as f:
            train_losses = json.load(f)
        with open(test_loss_path, "r") as f:
            test_losses = json.load(f)
        logger.info(f"Resumed loss history (train: {len(train_losses)}, test: {len(test_losses)})")
    else:
        logger.info(f"Did not resume loss history")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        # Training loop
        for lower, upper in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}"):
            lower, upper = lower.to(device), upper.to(device)

            optimizer.zero_grad()
            output = model(lower)
            loss = loss_fn(output, upper)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * lower.size(0)

        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Testing loss
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for lower, upper in test_loader:
                lower, upper = lower.to(device), upper.to(device)
                loss = loss_fn(model(lower), upper)
                epoch_test_loss += loss.item() * lower.size(0)

        test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(test_loss)

        # Log/plot stuff
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | LR: {learning_rate}")
        if not epoch % checkpoint_interval:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch{epoch}.pt")
        if not epoch % plot_interval:
            with open(train_loss_path, "w") as f:
                json.dump(train_losses, f)
            with open(test_loss_path, "w") as f:
                json.dump(test_losses, f)
            plot_losses(train_losses, test_losses, checkpoint_dir / "loss_curve.png")
            display_reconstructions(model, test_loader, device, checkpoint_dir / f"recon_epoch{epoch}.png")
        return
