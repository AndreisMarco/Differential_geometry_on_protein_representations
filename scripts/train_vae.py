import os
from functools import partial
import yaml

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from sklearn.cluster import KMeans 

from models import VAE_onehot, VAE_esm, VAE_adj_var 
from data_handling.onehot_dataset import int2onehot
from utils import data_loader, dataset_split, plot_loss, plot_latents, get_latent_representations

############ LOSS FUNCTIONS ############ 
def kl_divergence_with_standard_gaussian(mean, std):
    var = std**2
    kl_per_sample = 0.5 * jnp.sum(var + mean**2 - 1.0 - jnp.log(var), axis=-1)
    return kl_per_sample

def onehot_loss(model, x, rngs, kl_weight=1, inducing_points=None, use_var_adj=False):
    x_onehot = int2onehot(x, flatten=False)
    logits, mu_z, std_z = model(x=x_onehot, rngs=rngs, out_dim=x_onehot.shape)
    # Crossentropy reconstruction loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    recon_loss = -jnp.sum(x_onehot * log_probs, axis=-1).mean()
    kl_loss = kl_divergence_with_standard_gaussian(mu_z, std_z).mean()
    return recon_loss + kl_loss * kl_weight, (recon_loss, kl_loss * kl_weight)

def esm_loss(model, x, rngs, kl_weight=1, inducing_points=None, use_var_adj=False):
    mu_recon, std_recon, mu_z, std_z = model(x=x, rngs=rngs)
    log_prob = norm.logpdf(x=x, loc=mu_recon, scale=std_recon)
    recon_loss = -jnp.mean(log_prob, axis=-1).mean()
    kl_loss = kl_divergence_with_standard_gaussian(mu_z, std_z).mean()
    return recon_loss + kl_loss * kl_weight, (recon_loss,  kl_loss  * kl_weight)

def get_inducing_points(model, x, n_clusters=200):
    enc_out = model.encoder(x)
    mu_z, _ = jnp.split(enc_out, 2, axis=-1)
    kmeans = KMeans(n_clusters=n_clusters).fit(X=np.array(mu_z))
    return kmeans.cluster_centers_

def esm_var_adj_loss(model, x, rngs, inducing_points, kl_weight=1, use_var_adj=False):
    # Compute encoding (same for both branches)
    mu_z, std_z = model.encode(x)
    
    # Sample latent
    from models import rsample  # Assuming rsample is defined in models
    z = rsample(mu=mu_z, std=std_z, key=rngs.latent())
    
    # Decode with or without variance adjustment
    dec_out = model.decoder(z)
    mu_recon, log_var_recon = jnp.split(dec_out, 2, axis=-1)
    var_recon = jnp.exp(log_var_recon)
    
    # There is definetly a way to use the decoding function from the VAE, but I had problems with jitting the function across
    # python files so I had to give up and instead implement var adjustment here.
    
    # Apply variance adjustment if enabled
    def apply_var_adj(var_recon):
        z_input = z if z.ndim == 2 else z[None, :]
        dists_sq = jnp.sum((z_input[:, None, :] - inducing_points[None, :, :])**2, axis=-1)
        delta = jnp.sqrt(jnp.maximum(jnp.min(dists_sq, axis=-1), 1e-8))
        gamma_safe = jnp.maximum(jnp.abs(model.gamma), 1e-6)
        a = model.a_factor * gamma_safe
        nu = jax.nn.sigmoid((delta + a) / gamma_safe)
        var_adj = (1.0 - nu[:, None]) * var_recon + nu[:, None] * model.eta
        return var_adj
    
    var_final = jax.lax.cond(
        use_var_adj,
        apply_var_adj,
        lambda v: v,
        var_recon
    )
    
    std_recon = jnp.sqrt(var_final)
    
    log_prob = norm.logpdf(x=x, loc=mu_recon, scale=std_recon)
    recon_loss = -jnp.mean(log_prob, axis=-1).mean()
    kl_loss = kl_divergence_with_standard_gaussian(mu_z, std_z).mean()
    return recon_loss + kl_loss * kl_weight, (recon_loss,  kl_loss  * kl_weight)

############ TRAINING FUNCTIONS ############
@partial(nnx.jit, static_argnums=(1,))
def train_step(model, loss_fn, optimizer, x, rngs, kl_weight=1, inducing_points=None, use_var_adj=False):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon_loss, kl_loss)), grads = grad_fn(model=model, x=x, rngs=rngs, kl_weight=kl_weight, inducing_points=inducing_points, use_var_adj=use_var_adj)
    optimizer.update(grads)
    return loss, recon_loss, kl_loss

@partial(nnx.jit, static_argnums=(1,))
def test_step(model, loss_fn, x, rngs, kl_weight=1, inducing_points=None, use_var_adj=False):
    loss, (recon_loss, kl_loss) = loss_fn(model=model, x=x, rngs=rngs, kl_weight=kl_weight, inducing_points=inducing_points, use_var_adj=use_var_adj)
    return loss, recon_loss, kl_loss

def fit(model, loss_fn, optimizer, train_set, test_set, num_epochs, batch_size, kl_weight, rngs, switch_epoch=250):
    # Initialize loss bookeeping
    loss_history = {
        "train": {"loss": [], "recon_loss": [], "kl_loss": []},
        "test":  {"loss": [], "recon_loss": [], "kl_loss": []}}
    
    # Get initial inducing points (will be updated at switch_epoch)
    inducing_points = None
    
    for epoch in range(num_epochs):
        # Determine if we should use variance adjustment
        use_var_adj = epoch >= switch_epoch
        
        # Update inducing points at switch epoch
        if isinstance(model, VAE_adj_var) and epoch == switch_epoch:
            print(f"\nEpoch {epoch}: Switching to variance adjustment")
            inducing_points = get_inducing_points(model=model, x=train_set["x"], n_clusters=cfg["model"]["inducing_points"])
        
        # Get inducing points for each epoch if before switch or if not VAE_adj_var
        if isinstance(model, VAE_adj_var) and inducing_points is None:
            inducing_points = get_inducing_points(model=model, x=train_set["x"], n_clusters=cfg["model"]["inducing_points"])
        
        # Train loop
        cum_train = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        num_train_batches = 0
        for batched_data in data_loader(train_set, batch_size):
            loss, recon_loss, kl_loss = train_step(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                x=batched_data["x"],
                rngs=rngs,
                kl_weight=kl_weight[epoch],
                inducing_points=inducing_points,
                use_var_adj=use_var_adj)
            
            cum_train["loss"] += loss
            cum_train["recon_loss"] += recon_loss
            cum_train["kl_loss"] += kl_loss
            num_train_batches += 1

        # Test loop
        cum_test = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
        num_test_batches = 0
        for batched_data in data_loader(test_set, batch_size):
            loss, recon_loss, kl_loss = test_step(
                model=model,
                loss_fn=loss_fn,
                x=batched_data["x"],
                rngs=rngs,
                kl_weight=kl_weight[epoch],
                inducing_points=inducing_points,
                use_var_adj=use_var_adj)
            
            cum_test["loss"] += loss
            cum_test["recon_loss"] += recon_loss
            cum_test["kl_loss"] += kl_loss
            num_test_batches += 1

        # Log avg losses
        avg_train = {k: v / num_train_batches for k, v in cum_train.items()}
        avg_test = {k: v / num_test_batches for k, v in cum_test.items()}
        for key in ["loss", "recon_loss", "kl_loss"]:
            loss_history["train"][key].append(avg_train[key])
            loss_history["test"][key].append(avg_test[key])

        # Print epoch summary
        var_adj_status = f" [VAR_ADJ]" if use_var_adj else ""
        print(f"Epoch {epoch:3d}{var_adj_status} | "
              f"Train Loss: {avg_train['loss']:.4f} "
              f"(Recon: {avg_train['recon_loss']:.4f}, KL: {avg_train['kl_loss']:.4f}) | "
              f"Test Loss: {avg_test['loss']:.4f} "
              f"(Recon: {avg_test['recon_loss']:.4f}, KL: {avg_test['kl_loss']:.4f})")
        
    return model, loss_history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/train_vae.yaml', help='Path to yaml configuration file (default: %(default)s)')
    args = parser.parse_args()

    # Read configuration file
    cfg_path = args.cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # Create path to save model and logs
    save_path = cfg["save_path"]
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.abspath(save_path)

    # Initialize rngs
    split_key = jax.random.PRNGKey(cfg["seed"])
    seed_params, seed_latent = cfg["seed"], cfg["seed"] + 1
    rngs = nnx.Rngs(params=seed_params, latent=seed_latent)

    # Load data into dataset
    data = np.load(file=cfg["data_path"])
    dataset = {"x": data["embedding"],
               "y": data["label"],
               "padding_mask": data["padding_mask"]}
    
    # Average along sequence dimension
    if cfg["model"]["architecture"] != "vae_onehot" and cfg["data"]["avg_seq_len"]: 
        padding_mask = dataset["padding_mask"][..., None]
        dataset["x"] = np.mean(dataset["x"] * padding_mask, axis=1)

    print(f"Embeddings shape: {data['embedding'].shape}")
    print(f"Labels shape: {data['label'].shape}")

    # If not avg over seq dim, then flatten
    if cfg["model"]["architecture"] != "vae_onehot" and not cfg["data"]["avg_seq_len"]: 
        dataset["x"] = dataset["x"].reshape(dataset["x"].shape[0], -1)

    # Split in train and test set
    train_set, test_set = dataset_split(dataset=dataset, key=split_key)

    if cfg["model"]["architecture"] != "vae_onehot" and cfg["data"]["normalize"]:
        mean = train_set["x"].mean(axis=0)
        std = train_set["x"].std(axis=0) + 1e-8
        train_set["x"] = (train_set["x"] - mean) / std
        test_set["x"] = (test_set["x"] - mean) / std
        dataset["x"] = (dataset["x"] - mean) / std
    
    example_input = train_set["x"][:5]
    if cfg["model"]["architecture"] == "vae_onehot":
        example_input = int2onehot(example_input)
        example_input = example_input.reshape(5, -1)
    input_dim = example_input.shape[1]
    cfg["model"]["input_dim"] = input_dim
    print(f"Flattened input dim: {input_dim}")

    # Istantiate model
    model_dict = {"vae_onehot": VAE_onehot,
                  "vae_esm": VAE_esm,
                  "vae_adj_var": VAE_adj_var}
    VAE = model_dict[cfg["model"]["architecture"]]

    model = VAE(input_dim=input_dim,
                encoder_dim=cfg["model"]["encoder_dim"],
                decoder_dim=cfg["model"]["decoder_dim"],
                latent_dim=cfg["model"]["latent_dim"],
                rngs=rngs,
                gamma_init=cfg["model"]["gamma_init"],
                eta=cfg["model"]["eta"],
                a_factor=cfg["model"]["a_factor"])
    
    # Initialize optimizer
    num_batches_per_epoch = len(train_set["x"]) // cfg["train"]["batch_size"]
    total_steps = cfg["train"]["num_epochs"] * num_batches_per_epoch
    schedule = optax.cosine_decay_schedule(init_value=cfg["train"]["lr"], decay_steps=total_steps)
    optimizer = nnx.Optimizer(model,
                            optax.adamw(learning_rate=schedule, weight_decay=cfg["train"]["weight_decay"]),
                            wrt=nnx.Param)
    
    # Linear warmup for kl_loss
    kl_weight = np.ones(cfg["train"]["num_epochs"]) * cfg["train"]["kl_end"]
    kl_weight[:cfg["train"]["warmup_epochs"]] = np.linspace(cfg["train"]["kl_start"], 
                                                            cfg["train"]["kl_end"], 
                                                            cfg["train"]["warmup_epochs"])
    
    # Get switch epoch from config (default to 250 if not specified)
    switch_epoch = cfg["train"].get("var_adj_switch_epoch", 250)
    
    # Train model
    loss_dict = {"vae_onehot": onehot_loss,
                "vae_esm": esm_loss,
                "vae_adj_var": esm_var_adj_loss}
    model, loss_history = fit(model=model, 
                             loss_fn=loss_dict[cfg["model"]["architecture"]],
                             optimizer=optimizer,
                             train_set=train_set,
                             test_set=test_set,
                             num_epochs=cfg["train"]["num_epochs"],
                             batch_size=cfg["train"]["batch_size"],
                             kl_weight=kl_weight,
                             rngs=rngs,
                             switch_epoch=switch_epoch)
    
    # Save copy of train configs
    cfg_copy_path = os.path.join(save_path, "cfg.yaml")
    with open(cfg_copy_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    
    # Save model
    state_path = os.path.join(save_path, "state")
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(state_path, state)

    # Save loss plot
    loss_plot_file = os.path.join(save_path, "loss_history.png")
    plot_loss(loss_history, save_path=loss_plot_file)

    # Save latents
    # Both train and test separated for knn evaluation
    latent_file_train = os.path.join(save_path, "train_latents.npz")
    latent_file_test = os.path.join(save_path, "test_latents.npz")
    train_latents, train_labels = get_latent_representations(model=model, dataset=train_set, batch_size=cfg["train"]["batch_size"],
                                                             processing_fn=int2onehot if cfg["model"]["architecture"] == "vae_onehot" else None)
    np.savez(latent_file_train, embedding=train_latents, label=train_labels)
    test_latents, test_labels = get_latent_representations(model=model, dataset=test_set, batch_size=cfg["train"]["batch_size"],
                                                           processing_fn=int2onehot if cfg["model"]["architecture"] == "vae_onehot" else None)
    np.savez(latent_file_test, embedding=test_latents, label=test_labels)
    # And whole dataset in original order
    latent_file = os.path.join(save_path, "latents.npz")
    dataset
    latents, labels = get_latent_representations(model=model, dataset=dataset, batch_size=cfg["train"]["batch_size"],
                                                 processing_fn=int2onehot if cfg["model"]["architecture"] == "vae_onehot" else None)
    np.savez(latent_file, embedding=latents, label=labels)
    
    # Save latent plot
    latent_plot_file = os.path.join(save_path, "latents.png")
    plot_latents(latents=latents,
                 labels=labels, 
                 save_path=latent_plot_file)
    print(f"Trained model and logs saved at {save_path}")
