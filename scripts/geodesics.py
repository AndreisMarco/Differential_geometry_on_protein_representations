import os
from itertools import combinations
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

from sklearn.cluster import KMeans

from utils import load_VAE, plot_latents

class Geodesics():
    def __init__(self, model,
                 num_points,
                 latent_dim,
                 inducing_points,
                 key):
        self.model = model
        self.inducing_points = inducing_points

        def decoder_mean(z):
            mu_recon, std_recon = model.decode(z, inducing_points)
            return mu_recon  # Only return mean
        self.jacobian_mean = jax.jacfwd(decoder_mean)
        def decoder_std(z):
            mu_recon, std_recon = model.decode(z, inducing_points)
            return std_recon  # Only return std
        self.jacobian_std = jax.jacfwd(decoder_std)

        self.latent_dim = latent_dim
        self.num_points = num_points
        self.key = key
        self.c_t = None

    def init_curves(self, extremes, sigma=0.01):
        """
        Initialize curves between point pairs as straight lines
        """
        t = jnp.linspace(0, 1, self.num_points) 
        starts = extremes[:, 0, :]  
        ends = extremes[:, 1, :]
        cum_increments = t[None, :, None] * (ends - starts)[:, None, :]
        straight_lines = starts[:, None, :] + cum_increments

        # Add random noise to initialized straight lines
        if sigma is not None and sigma > 0:
            self.key, subkey = jax.random.split(self.key)
            noise = sigma * jax.random.normal(subkey, straight_lines.shape)
            noise = noise.at[:, 0, :].set(0.0)
            noise = noise.at[:, -1, :].set(0.0)
            straight_lines += noise
        self.c_t = straight_lines
    
    def compute_metric(self, c_t):
        """
        Compute riemannian metric as define by Arvanatidis et al. 2017
        """
        J_mean = jax.vmap(self.jacobian_mean)(c_t)
        J_mean = J_mean.reshape(c_t.shape[0], -1, self.latent_dim)
        J_std = jax.vmap(self.jacobian_std)(c_t)
        J_std = J_std.reshape(c_t.shape[0], -1, self.latent_dim)
        G = jnp.einsum('ndi,ndj->nij', J_mean, J_mean) + \
            jnp.einsum('ndi,ndj->nij', J_std, J_std)
        # Normalize by data dimension
        G = G / J_mean.shape[1]
        return G
    
    def curve_length(self, c_t):
        '''
        Compute lengths of curves as sqrt of energy
        '''
        delta = c_t[1:] - c_t[:-1] # (num_points-1, latent_dim)
        midpoints = (c_t[1:] + c_t[:-1]) / 2
        metric = self.compute_metric(midpoints) # (num_points-1, latent_dim, latent_dim)
        energy_points = jnp.einsum('ni,nij,nj->n', delta, metric, delta) # (num_points-1, )
        return jnp.sum(jnp.sqrt(energy_points + 1e-12))
    
    def curve_energy_metric(self, c_t):
        '''
        Compute energy of curves using the pullback metric
        '''
        delta = c_t[1:] - c_t[:-1] # (num_points-1, latent_dim)
        middle_points = (c_t[1:] + c_t[:-1]) / 2
        metric = self.compute_metric(middle_points) # (num_points-1, latent_dim, latent_dim)
        energy_points = jnp.einsum('ni,nij,nj->n', delta, metric, delta) # (num_points-1, )
        energy = energy_points.sum() # (), i.e. scalar
        return energy
    
    def curve_energy_bruteforce(self, c_t):
        '''
        Compute energy of curves projecting points in the data space
        '''
        f_c = jax.vmap(self.model.decoder)(c_t)  # (num_points, data_dim)
        diff = f_c[1:] - f_c[:-1]  # (num_points-1, data_dim)
        energy = jnp.sum(diff**2) # (), i.e. scalar
        return energy
    

    def optimize_geodesics(self, energy_fn="metric", num_steps=100, lr=0.001):
        '''
        Find the geodesics by minimizing the curve energy
        '''
        c_t = self.c_t
        if energy_fn == "bruteforce":
            loss_fn = self.curve_energy_bruteforce
        elif energy_fn == "metric":
            loss_fn = self.curve_energy_metric

        energy_history = []
        # vectorize over num_geodesics dim, basically a batch_dim
        energy_and_grads_batched = lambda curves: jax.vmap(jax.value_and_grad(loss_fn))(curves) 
        progress_bar = tqdm(range(num_steps), desc="Optimizing curves")
        # Istantiate optimizer 
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=num_steps)
        opt = optax.adamw(learning_rate=schedule)
        opt_state = opt.init(c_t)
        # Optimize points coordinates
        for step in range(num_steps):
            energy, grads = energy_and_grads_batched(c_t)
            # fix end points
            grads = grads.at[:, 0, :].set(0.0)
            grads = grads.at[:, -1, :].set(0.0)
            updates, opt_state = opt.update(grads, opt_state, params=c_t)
            c_t = optax.apply_updates(c_t, updates)
            energy_history.append(energy)
            progress_bar.set_postfix(avg_energy=f"⠀{energy.mean().item():.4f}", step=f"{step+1}/{num_steps}")
            progress_bar.update()
        energy_history = jnp.vstack(energy_history).T # traspose to keep consistency of each row = different geodesic

        # Compute energy and length of optimized curves
        final_energy = jax.vmap(self.curve_energy_metric)(c_t)
        final_length = jax.vmap(self.curve_length)(c_t)
        self.c_t = c_t
        return c_t, energy_history, final_energy, final_length
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/geodesics/geodesics.yaml', help='Path to yaml configuration file (default: %(default)s)')
    args = parser.parse_args()

    # Read configuration file
    cfg_path = args.cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Load model
    np.random.seed(cfg["seed"])
    rngs = nnx.Rngs(cfg["seed"])
    model_path = cfg["model_path"]
    model, model_cfg = load_VAE(rngs=rngs, model_path=model_path)

    # Create directory to store logs and geodesics
    geodesics_dir = os.path.join(model_path, "geodesics")
    os.makedirs(geodesics_dir, exist_ok=True)

    # Load latents
    def load_latent(file):
        data = np.load(file)
        latents = data["embedding"]
        labels = data["label"]
        return latents, labels
    
    latent_file = os.path.join(model_path, "latents.npz")
    latents, labels = load_latent(latent_file)
    train_latent_file = os.path.join(model_path, "train_latents.npz")
    train_latents, _ = load_latent(latent_file)

    # Select closest point to each centroid of the train latents
    kmeans = KMeans(n_clusters=cfg["num_considered_latents"]).fit(train_latents)
    centroids = kmeans.cluster_centers_    
    distances = ((latents[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2).sum(axis=2)
    selected_idx = closest_indices = np.argmin(distances, axis=0) 
    selected_latents = latents[selected_idx]
    selected_labels = labels[closest_indices]

    # Plot selected points over latents
    selected_file = os.path.join(geodesics_dir, "selected_points.pdf")
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax = plot_latents(latents=latents, labels=labels, ax=ax, legend=True)
    ax.scatter(selected_latents[:,0], selected_latents[:,1], marker="x",
               s=20, c="k", linewidth=1.2)
    plt.savefig(selected_file, format="pdf", bbox_inches='tight')

    # Assemble all pairs of start-end latents
    pairs = np.array(list(combinations(selected_idx, 2)))
    num_geodesics = pairs.shape[0]
    print(f"Total number of geodesics: {num_geodesics}")

    # print("!!! DEBUGGING ON ONLY 256 GEODESICS!!!")
    # pairs = pairs[np.random.choice(num_geodesics, size=256)]
    # num_geodesics = pairs.shape[0]

    # Initialize optimizer
    geodesics_optimizer = Geodesics(model=model,
                          num_points=cfg["optimization"]["num_points"],
                          latent_dim=model_cfg["model"]["latent_dim"],
                          inducing_points=centroids,
                          key=rngs())
    
    # Initialize storing arrays
    geodesics_all = np.zeros(shape=(num_geodesics, cfg["optimization"]["num_points"], model_cfg["model"]["latent_dim"]))
    energy_history_all = np.zeros(shape=(num_geodesics, cfg["optimization"]["num_steps"]))
    final_energy_all = np.zeros(shape=(num_geodesics,)) 
    final_length_all = np.zeros(shape=(num_geodesics,)) 
    euclidean_dist = np.zeros(shape=(num_geodesics,)) 

    # Batch geodesic optimization
    batch_size = cfg["optimization"]["batch_size"]
    print("Starting optimization")
    for i in range(0, num_geodesics, batch_size):
        idx_batch = pairs[i:i+batch_size]
        curr_batch_size = idx_batch.shape[0]
        latents_pairs = latents[idx_batch]
        geodesics_optimizer.init_curves(latents_pairs)
        geodesics, energy_history, final_energy, final_length = geodesics_optimizer.optimize_geodesics(lr=cfg["optimization"]["lr"],
                                                                                                        energy_fn="metric",
                                                                                                        num_steps=cfg["optimization"]["num_steps"])
        geodesics_all[i:i+curr_batch_size]      = geodesics
        energy_history_all[i:i+curr_batch_size] = energy_history
        final_energy_all[i:i+curr_batch_size]   = final_energy
        final_length_all[i:i+curr_batch_size]   = final_length
        euclidean_dist[i:i+curr_batch_size] = np.linalg.norm(latents_pairs[:, 1] - latents_pairs[:, 0], axis=1)
    
    # Save optimized geodesics
    geodesics_file = os.path.join(geodesics_dir, "geodesics.npz")
    np.savez(geodesics_file,
             idx=pairs,
             geodesics=geodesics_all,
             length=final_length_all,
             energy=final_energy_all,
             euclidean_dist=euclidean_dist)
    # Save energy history
    energy_history_file = os.path.join(geodesics_dir, "energy_history.npz")
    np.savez(energy_history_file,
             energy_history=energy_history_all)

