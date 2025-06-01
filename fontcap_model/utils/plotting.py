from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn import Module
from torch import no_grad

"""
Utility functions to plot stuff during or after training
"""

def plot_losses(
        train_losses: list[float], 
        test_losses: list[float], 
        path: Path | None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.close()
    return

def display_reconstructions(
        model: Module, 
        dataloader, 
        device, 
        path: Path | None, 
        num_images: int = 8):
    """Displays actual vs predicted model output images"""
    model.eval()
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(device), targets.to(device)

    with no_grad():
        outputs = model(inputs)

    inputs = inputs[:num_images].cpu().squeeze(1).numpy()
    outputs = outputs[:num_images].cpu().squeeze(1).numpy()
    targets = targets[:num_images].cpu().squeeze(1).numpy()

    _, axes = plt.subplots(num_images, 3, figsize=(6, num_images * 1.5))
    for i in range(num_images):
        axes[i, 0].imshow(inputs[i], cmap="gray")
        axes[i, 0].set_title("input")
        axes[i, 1].imshow(outputs[i], cmap="gray")
        axes[i, 1].set_title("output")
        axes[i, 2].imshow(targets[i], cmap="gray")
        axes[i, 2].set_title("target")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.close()
    return
