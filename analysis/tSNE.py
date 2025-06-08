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


def plot_tsne_2d(latents, colours=None, sizes=5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111,)
    ax.scatter(latents[:, 0], latents[:, 1], c=colours, cmap='tab20', s=sizes, alpha=0.7)
    ax.set_title('t-SNE Projection of Latents')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)
    return fig


def plot_tsne_3d(latents, colours=None, sizes=5):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        latents[:, 0], latents[:, 1], latents[:, 2],
        c=colours, cmap='tab20', s=sizes, alpha=0.7
    )
    ax.set_title('t-SNE Projection of Latents')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    return fig


def plot_tsne_2d_restricted(
        latents,
        latent_labels,  # The labels attached to each latent
        labels_to_plot,  # Which labels to include in the plot
        sizes=5
):
    """Plots latents with labels"""
    # Indexes of latents/labels to plot
    idx = [i for i, label in enumerate(latent_labels) if label in labels_to_plot]
    allowed_latents = latents[idx, :]
    allowed_latent_labels = [label for i, label in enumerate(latent_labels) if i in idx] # ['a', 'b']
    # Get the colours
    colour_encoding = [labels_to_plot.index(label) for label in allowed_latent_labels] # [0, 1]
    colours = plt.cm.tab20(colour_encoding)
    # Get each distinct label colour for the legend
    distinct_idx = [allowed_latent_labels.index(label) for label in labels_to_plot]
    legend_elts = [Patch(facecolor=colours[i], label=allowed_latent_labels[i]) for i in distinct_idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(allowed_latents[:, 0], allowed_latents[:, 1], c=colours, s=sizes, alpha=0.7)
    ax.set_title('t-SNE Projection of Latents')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(handles=legend_elts, title="Legend")
    ax.grid(True)
    return fig
