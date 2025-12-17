import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import flax.nnx as nnx
from utils import load_VAE, accumulated_variance, plot_fn_on_grid, plot_latents, metric_determinant
from sklearn.cluster import KMeans

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

# Configuration
cfg_path = 'configs/train_vae.yaml'
with open(cfg_path, 'r') as file:
    cfg = yaml.safe_load(file)

# Load model
np.random.seed(cfg["seed"])
rngs = nnx.Rngs(cfg["seed"])
model_path = cfg["save_path"]
model, model_cfg = load_VAE(rngs=rngs, model_path=model_path)

latents, labels, train_latents = load_latents_data(model_path=model_path)
inducing_points = None
if cfg["model"]["architecture"] == "vae_adj_var":
     kmeans = KMeans(n_clusters=model_cfg["model"]["inducing_points"]).fit(train_latents)
     inducing_points = kmeans.cluster_centers_

# Plot accumulated variance
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
decoding_fn = lambda z: model.decode(z=z, inducing_points=inducing_points)
variance_fn = lambda z: accumulated_variance(decoding_fn=decoding_fn, z=z, log=True)
ax = plot_fn_on_grid(fn=variance_fn, label="Log accumulated variance",
                     xlim=(-4, 4), ylim=(-4,4), res=50, cmap="inferno",
                     ax=ax)
ax = plot_latents(latents=latents, labels=labels, ax=ax, legend=False, alpha=1)
plt.tight_layout()
variance_plot_file = os.path.join(model_path, "acc_variance.pdf")
plt.savefig(variance_plot_file,  format="pdf", bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
# Plot metric determinant
decoding_fn = lambda z: model.decode(z=z, inducing_points=inducing_points)
metric_fn = lambda z: metric_determinant(decoding_fn=decoding_fn, z=z, log=True)
ax = plot_fn_on_grid(fn=metric_fn, label="Log metric determinant",
                    xlim=(-4, 4), ylim=(-4, 4), res=50, cmap="inferno",
                    ax=ax)
ax = plot_latents(latents=latents, labels=labels, ax=ax, legend=False, alpha=1)
metric_plot_file = os.path.join(model_path, "metric.pdf")
plt.savefig(metric_plot_file, format='pdf', bbox_inches='tight')
