

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import flax.nnx as nnx
from sklearn.cluster import KMeans

from utils import load_VAE, plot_latents, plot_fn_on_grid, metric_determinant

def load_geodesics_data(geodesics_dir):
    geodesics_file = os.path.join(geodesics_dir, "geodesics.npz")
    data = np.load(geodesics_file)
    return {
        'geodesics': data['geodesics'],
        'idx': data['idx'],
        'length': data['length'],
        'energy': data['energy']}

def load_latents_data(model_path):
    """Load latent representations and labels."""
    def load_split(split):
        latent_file = os.path.join(model_path, f"{split}_latents.npz")
        data = np.load(latent_file)
        return data["embedding"], data["label"]
    train_latents, train_labels = load_split("train")
    test_latents, test_labels = load_split("test")
    latents = np.concatenate([train_latents, test_latents], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)
    return latents, labels, train_latents

def plot_energy_history(energy_history_all, save_path):
    # Normalize energy histories to their initialization to ease visualization
    energy_history = energy_history_all / energy_history_all[:, 0][:, None]
    num_steps = energy_history.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Energy history (relative to straight lines)")
    for i in range(energy_history.shape[0]):
        ax.plot(range(num_steps), energy_history[i], c="k", alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Relative energy")
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved energy history plot at: {save_path}")

def plot_geodesics_on_latents(geodesics, latents, labels, save_path, alpha, num_geodesics, seed=42):
    np.random.seed(seed)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = plot_latents(latents=latents, labels=labels, ax=ax, legend=True)

    # Calculate how many geodesics to plot
    num_geodesics_total = geodesics.shape[0]
    num_to_plot = min(num_geodesics, num_geodesics_total) 
    # Randomly select indices
    idx = np.random.choice(num_geodesics_total, num_to_plot, replace=False)
    
    ax.plot(geodesics[idx, :, 0].T, geodesics[idx, :, 1].T, c="k", linewidth=1, alpha=alpha)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {num_to_plot} geodesics on latents plot to {save_path}")

def plot_geodesics_on_metric(geodesics, centroids, model, save_path, 
                              xlim=(-8, 8), ylim=(-8, 8), alpha=0.5, num_geodesics=256, seed=42):
    np.random.seed(seed)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot metric determinant
    decoding_fn = lambda z: model.decode(z=z, inducing_points=centroids)
    metric_fn = lambda z: metric_determinant(decoding_fn=decoding_fn, z=z, log=True)
    ax = plot_fn_on_grid(fn=metric_fn, label="Log metric determinant",
                        xlim=xlim, ylim=ylim, res=100, cmap="inferno", ax=ax)
    
    # Plot geodesics
    num_geodesics_total = geodesics.shape[0]
    num_to_plot = min(num_geodesics, num_geodesics_total) 
    # Randomly select indices
    idx = np.random.choice(num_geodesics_total, num_to_plot, replace=False)
    
    ax.plot(geodesics[idx, :, 0].T, geodesics[idx, :, 1].T, c="k", linewidth=1, alpha=alpha)
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {num_to_plot} geodesics on metric plot at: {save_path}")

def create_all_visualizations(geodesics_dir, model_path, model, alpha, num_geodesics, seed):
    # Load optimized geodesics
    geodesic_data = load_geodesics_data(geodesics_dir)
    geodesics = geodesic_data['geodesics']

    # Load energy history
    energy_history_file = os.path.join(geodesics_dir, "energy_history.npz")
    energy_history = np.load(energy_history_file)["energy_history"]

    # Load latents and compute centroids/inducing points
    latents, labels, train_latents = load_latents_data(model_path)
    kmeans = KMeans(n_clusters=cfg["num_considered_latents"]).fit(train_latents)
    centroids = kmeans.cluster_centers_
    
    plot_energy_history(
        energy_history, 
        os.path.join(geodesics_dir, "energy_history.pdf"))
    
    plot_geodesics_on_latents(
        geodesics, latents, labels,
        os.path.join(geodesics_dir, "geodesics_with_latents.pdf"),
        alpha=alpha,
        num_geodesics=num_geodesics,
        seed=seed)
    
    plot_geodesics_on_metric(
        geodesics, centroids, model,
        os.path.join(geodesics_dir, "geodesics_with_metric.pdf"),
        alpha=alpha,
        num_geodesics=num_geodesics,
        seed=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/geodesics/geodesics.yaml', help='Path to yaml configuration file')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for geodesic lines')
    parser.add_argument('--num_geodesics', type=int, default=256, help='Number of geodesics to plot randomly')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.cfg, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Setup paths
    model_path = cfg["model_path"]
    geodesics_dir = os.path.join(model_path, "geodesics")
    
    # Load model
    np.random.seed(cfg["seed"])
    rngs = nnx.Rngs(cfg["seed"])
    model, model_cfg = load_VAE(rngs=rngs, model_path=model_path)
    
    # Generate all visualizations
    create_all_visualizations(geodesics_dir, model_path, model, alpha=args.alpha, 
                              num_geodesics=args.num_geodesics, seed=cfg["seed"])
