import numpy as np
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

"""
Helpers related to t-SNE
"""


def run_tsne(
        latents,
        dim: int = 2,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int | None = None):
    """Wrapper for tSNE fit_transform"""
    if not random_state:
        random_state = randint(1, 100)
    tsne = TSNE(n_components=dim, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    return tsne.fit_transform(latents)


def plot_tsne_2d(latents, colours=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.axes[0]
    ax.scatter(latents[:, 0], latents[:, 1], c=colours, cmap='tab20', s=5, alpha=0.7)
    ax.title('t-SNE Projection of Latents')
    ax.xlabel('Feature 1')
    ax.ylabel('Feature 2')
    ax.grid(True)
    ax.tight_layout()
    return fig


def plot_tsne_3d(latents, colours=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        latents[:, 0], latents[:, 1], latents[:, 2],
        c=colours, cmap='tab20', s=8, alpha=0.7
    )
    ax.set_title('t-SNE Projection of Latents')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.tight_layout()
    return fig


def plot_tsne_2d_restricted(
        latents,
        latent_labels,  # The labels attached to each latent
        labels_to_plot  # Which labels to include in the plot
):
    """Plots latents with labels"""
    # Indexes of latents/labels to plot
    idx = [i for i, label in enumerate(latent_labels) if label in labels_to_plot]
    # Restrict the data
    labels = [l for i, l in enumerate(latent_labels) if i in idx]
    allowed_latents = latents[idx, :]

    # Get the colours
    colours = plt.cm.tab20([cl for cl in idx])

    # Get each distinct label colour for the legent
    distinct_idx = [labels.index(label) for label in labels_to_plot]
    legend_elts = [Patch(facecolor=colours[i], label=labels[i]) for i in distinct_idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.axes[0]
    ax.scatter(allowed_latents[:, 0], allowed_latents[:, 1], c=colours, s=5, alpha=0.7)
    ax.title('t-SNE Projection of Latents')
    ax.xlabel('Feature 1')
    ax.ylabel('Feature 2')
    ax.legend(handles=legend_elts, title="Legend")
    ax.grid(True)
    ax.tight_layout()
    return
