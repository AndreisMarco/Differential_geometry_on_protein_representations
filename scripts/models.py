import jax
import jax.numpy as jnp
import flax.nnx as nnx

def rsample(mu, std, key):
    '''
    Sampling with reparametrization trick
    '''
    eps = jax.random.normal(key=key, shape=mu.shape)
    z = mu + std * eps
    return z

class MLP(nnx.Module):
    def __init__(self,
                 input_dim: int, 
                 layer_dim: list[int], 
                 out_dim: int,
                 rngs: nnx.Rngs):
        
        self.linear1 = nnx.Linear(in_features=input_dim, out_features=layer_dim[0], rngs=rngs)
        self.linear2 = nnx.Linear(in_features=layer_dim[0], out_features=layer_dim[1], rngs=rngs)
        self.linear3 = nnx.Linear(in_features=layer_dim[1], out_features=layer_dim[2], rngs=rngs)
        self.linear4 = nnx.Linear(in_features=layer_dim[2], out_features=out_dim, rngs=rngs)
        self.activation_fn = nnx.gelu

    def __call__(self, x):
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        x = self.activation_fn(self.linear3(x))
        x = self.linear4(x)
        return x
    
class VAE_base(nnx.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_dim: list[int],
                 decoder_dim: list[int],
                 latent_dim: int,
                 rngs: nnx.Rngs,
                 **kwargs):
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim=input_dim,
                           layer_dim=encoder_dim,
                           out_dim=latent_dim*2,
                           rngs=rngs)
        
        decoder_out_dim = self.decoder_out_dim(input_dim)
        self.decoder = MLP(input_dim=latent_dim,
                           layer_dim=decoder_dim,
                           out_dim=decoder_out_dim,
                           rngs=rngs)
        
    def decoder_out_dim(self):
        # To overwrite
        pass

    def encode(self, x):
        x = x.reshape(x.shape[0], -1) # Flatten input 
        enc_out = self.encoder(x)
        mu_z, log_var_z = jnp.split(ary=enc_out, indices_or_sections=2, axis=-1)
        std_z = jnp.exp(0.5 * log_var_z)
        return mu_z, std_z

    def __call__(self):
        # To overwrite
        pass

class VAE_onehot(VAE_base):
    def decoder_out_dim(self, input_dim):
        return input_dim
    
    def __call__(self, x, rngs, out_dim=None, inducing_points=None):
        mu_z, std_z = self.encode(x)
        z = rsample(mu=mu_z, std=std_z, key=rngs.latent())
        logits = self.decoder(z)
        if out_dim is not None:
            logits = logits.reshape(out_dim)
        return logits, mu_z, std_z

class VAE_esm(VAE_base):
    def decoder_out_dim(self, input_dim):
        return input_dim * 2
    
    def decode(self, z, inducing_points=None):
        dec_out = self.decoder(z)
        mu_recon, log_var_recon = jnp.split(dec_out, 2, axis=-1)
        std_recon = jnp.exp(0.5 * log_var_recon)
        return mu_recon, std_recon
    
    def __call__(self, x, rngs, inducing_points=None):
        mu_z, std_z = self.encode(x)
        z = rsample(mu=mu_z, std=std_z, key=rngs.latent())
        mu_recon, std_recon = self.decode(z)
        return mu_recon, std_recon, mu_z, std_z
    
class VAE_adj_var(VAE_esm):
    def __init__(self, 
                 input_dim: int,
                 encoder_dim: list[int],
                 decoder_dim: list[int],
                 latent_dim: int,
                 rngs: nnx.Rngs,
                 gamma_init: float = 1.0,
                 eta: float = 100.0,
                 a_factor: float=-6.9077):
        super().__init__(input_dim=input_dim,
                         encoder_dim=encoder_dim,
                         decoder_dim=decoder_dim,
                         latent_dim=latent_dim,
                         rngs=rngs)
        self.eta = eta
        self.a_factor = a_factor
        self.gamma = nnx.Param(jnp.array(gamma_init))

    def __call__(self, x, inducing_points, rngs):
        mu_z, std_z = self.encode(x)
        z = rsample(mu=mu_z, std=std_z, key=rngs.latent())
        mu_recon, std_recon = self.decode(z, inducing_points)
        return mu_recon, std_recon, mu_z, std_z

    def decode(self, z, inducing_points):
        # Add batch dimension if necessary
        z_input = z if z.ndim == 2 else z[None, :]
        # Get decoder output
        dec_out = self.decoder(z_input)
        mu_recon, log_var_recon = jnp.split(dec_out, 2, axis=-1)
        
        # Clamp log_var to prevent extreme values
        var_recon = jnp.exp(log_var_recon)
                            
        # Find distance to closest inducing point
        dists_sq = jnp.sum((z_input[:, None, :] - inducing_points[None, :, :])**2, axis=-1)
        delta = jnp.sqrt(jnp.maximum(jnp.min(dists_sq, axis=-1), 1e-8))
        
        # Scale sigmoid
        gamma_safe = jnp.maximum(jnp.abs(self.gamma), 1e-6)
        a = self.a_factor * gamma_safe
        
        # Adjust variance
        nu = jax.nn.sigmoid((delta + a) / gamma_safe)
        var_adj = (1.0 - nu[:, None]) * var_recon + nu[:, None] * self.eta
        std_recon = jnp.sqrt(var_adj)

        if z.ndim == 1:
            mu_recon = mu_recon[0]
            std_recon = std_recon[0]
        
        return mu_recon, std_recon