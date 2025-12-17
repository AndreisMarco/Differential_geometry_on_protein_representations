import os
import yaml

import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp

import flax.nnx as nnx
import orbax.checkpoint as ocp

from sklearn.decomposition import PCA

from models import VAE_onehot, VAE_esm, VAE_adj_var

######### DATA HANDLING ######### 
def dataset_split(dataset, key, train_split=0.8):
    '''
    Splits the dataset in training and test set
    '''
    # Check consistent number of datapoints
    n_datapoints = [x.shape[0] for x in dataset.values()]
    assert len(set(n_datapoints)) == 1, f"Mismatch in dataset lengths: {n_datapoints}"

    N = n_datapoints[0]
    perm = jax.random.permutation(key, N)
    split_idx = int(train_split * N)
    dataset_splitted = {"train": {}, "test": {}}
    for k in dataset.keys():
        dataset_splitted["train"][k] = dataset[k][perm][:split_idx]
        dataset_splitted["test"][k] = dataset[k][perm][split_idx:]
    return dataset_splitted["train"], dataset_splitted["test"]

def data_loader(dataset, batch_size):
    '''
    Provides batches of the dataset to iterate over
    '''
    N = len(dataset["x"])
    idxs = np.arange(N)
    for i in range(0, N, batch_size):
        batch_idxs = idxs[i:i + batch_size]
        yield {
            "x": dataset["x"][batch_idxs],
            "y": dataset["y"][batch_idxs]}

######### PLOTTING ######### 
def plot_loss(loss_history, show=False, save_path=None):
    epochs = range(0, len(loss_history["train"]["loss"]))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, loss in enumerate(loss_history["train"].keys()):
        axs[i].plot(epochs, loss_history["train"][loss], label="Train")
        axs[i].plot(epochs, loss_history["test"][loss], label="Test")
        axs[i].set_title(loss)
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].grid(True)
        axs[i].legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def plot_latents(latents,
                 labels, 
                 alpha=0.5,
                 title=None, 
                 legend=True,
                 ax=None,
                 save_path=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    latent_dim = latents.shape[1]
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latents = pca.fit_transform(latents)
        label_x = "PC1"
        label_y = "PC2"
    else:
        label_x = r"$\mu_1$"
        label_y = r"$\mu_2$"

    unique_classes = np.unique(labels)
    for cls in unique_classes:
        mask = labels == cls
        ax.scatter(latents[mask, 0], latents[mask, 1], s=10, alpha=alpha, edgecolors="k", linewidth=0.1, label=cls)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(title='Caspase', bbox_to_anchor=(1.05, 1), loc='upper left')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return ax

def plot_fn_on_grid(fn, xlim, ylim, res, label, cmap='Greys_r', ax=None, save_path=None):
    # Create grid
    x = np.linspace(*xlim, res)
    y = np.linspace(*ylim, res)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=-1)

    # Evaluate fn at each point
    Z = jax.vmap(jax.vmap(fn))(grid)
    Z = np.array(Z)
    
    # Plot fn values
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.set(xlabel='z₁', ylabel='z₂', xlim=xlim, ylim=ylim, aspect='equal')
    vmin, vmax = np.percentile(Z, [5, 95])
    contour = ax.contourf(X, Y, np.clip(Z, vmin, vmax), levels=50, cmap=cmap)
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.03, location='top', shrink=0.8)
    cbar.set_label(label)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return ax

######### OTHERS ######### 
def load_VAE(model_path, rngs):
    # Load model configuration
    cfg_path = os.path.join(model_path, "cfg.yaml")
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # Istantiate dummy model
    model_dict = {"vae_onehot": VAE_onehot,
                  "vae_esm": VAE_esm,
                  "vae_adj_var": VAE_adj_var}
    VAE = model_dict[cfg["model"]["architecture"]]
    model = VAE(input_dim=cfg["model"]["input_dim"],
                encoder_dim=cfg["model"]["encoder_dim"],
                decoder_dim=cfg["model"]["decoder_dim"],
                latent_dim=cfg["model"]["latent_dim"],
                rngs=rngs,
                gamma_init=cfg["model"]["gamma_init"],
                eta=cfg["model"]["eta"],
                a_factor=cfg["model"]["a_factor"])
    
    graphdef, _ = nnx.split(model)
    # Load state in the scaffold model
    checkpointer = ocp.StandardCheckpointer()
    state_path = os.path.abspath(os.path.join(model_path, "state"))
    restored_state_dict = checkpointer.restore(state_path) 
    model = nnx.merge(graphdef, restored_state_dict)
    return model, cfg

def get_latent_representations(model, dataset, batch_size, processing_fn=None, include_std=False):
    # Create array to hold the representation
    if include_std:
        latent_dim = model.latent_dim*2
    else:
        latent_dim = model.latent_dim
    num_datapoints = dataset["x"].shape[0]
    latent_array = np.zeros(shape=(num_datapoints, latent_dim))

    # Compute latent representations
    i = 0
    for batched_data in data_loader(dataset, batch_size):
        x = batched_data["x"]
        curr_batch_size = x.shape[0]
        if processing_fn is not None:
            x = processing_fn(x)
        mu, std = model.encode(x=x)
        mu, std = np.array(mu), np.array(std)
        if include_std:
            latent_array[i:i+curr_batch_size] = np.concatenate([mu, std], axis=1)
        else:
            latent_array[i:i+curr_batch_size] = mu
        i += curr_batch_size
    return latent_array, dataset["y"]

def metric_determinant(decoding_fn, z, log=False):
    J_mean, J_std = jax.jacfwd(decoding_fn)(z)  # (data_dim, latent_dim)
    G = J_mean.T @ J_mean + J_std.T @ J_std  # (latent_dim, latent_dim)
    sign, logdet = jnp.linalg.slogdet(G)
    if log:
        return logdet / 2
    else:
        return jnp.sqrt(jnp.exp(logdet))
        
def metric_fn(z, decoding_fn, inducing_points):
    def get_mean(z_in):
        return decoding_fn(z_in, inducing_points)[0].flatten()
    
    def get_std(z_in):
        return decoding_fn(z_in, inducing_points)[1].flatten()

    J_mean = jax.jacrev(get_mean)(z)
    J_std = jax.jacrev(get_std)(z)
    G = jnp.einsum("oi,oj->ij", J_mean, J_mean) + jnp.einsum("oi,oj->ij", J_std, J_std)
    output_dim = J_mean.shape[0]
    G = G / J_mean.shape[1]
    return G    

def accumulated_variance(decoding_fn, z, log=False):
    _, log_var_recon = decoding_fn(z)
    var_recon = jnp.exp(log_var_recon)
    acc_var = jnp.sum(var_recon)
    if log:
        return jnp.log(acc_var)
    else:
        return acc_var
