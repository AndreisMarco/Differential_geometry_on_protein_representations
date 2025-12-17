import pandas as pd
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp

from flax.core import frozen_dict
from esmjax import io, tokenizer
from esmjax.modules import modules

import argparse
parser = argparse.ArgumentParser(description="Encode proteins using esmjax")
parser.add_argument("-input_file", type=str, default="0_reset/data/caspase_info_filtered.csv", help="Path to the file containing the sequences")
parser.add_argument("-esm_model", type=str, choices=("esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D"), default="esm2_t12_35M_UR50D", help="ESM model to be used for the embeddings")
parser.add_argument("-filter_family", type=int , default=0, help="Which caspase family to encode (if 0 encode all)")
parser.add_argument("-out_file", type=str, default="0_reset/data/esm_representation_480d.npz", help="Output file where to save the embeddings")
args = parser.parse_args()

# Download model weights
model_name = args.esm_model
print(f"Using model: {model_name}")
state = io.get_torch_state(model_name)
# Convert to jax standard params dictionary
esm_params = io.convert_encoder(state["model"], state["cfg"]) 
esm_params = frozen_dict.FrozenDict({"params": esm_params})
# Istantiate model architecture 
esm, _ = modules.get_esm2_model(state["cfg"])

# Extract sequences and caspase id from input file
df = pd.read_csv(args.input_file) 

if args.filter_family != 0:
    df = df[df["caspase"] == args.filter_family]

sequences = df["sequence"].tolist()
caspase_ids = df["caspase"].to_numpy()
num_sequences = len(sequences)
print(f"Found {num_sequences} sequences")

# Tokenize and extract ids 
print("Tokenizing sequences")
tokenizer = tokenizer.protein_tokenizer(pad_to_multiple_of=128)
encodings = tokenizer.encode_batch(sequences)
encodings_ids = jnp.array(np.stack([enc.ids for enc in encodings]))
padding_mask = jnp.array(np.stack([enc.attention_mask for enc in encodings])) 

# Build array to store embeddings
padded_seq_len = encodings_ids.shape[1]
if model_name ==  "esm2_t12_35M_UR50D":
    embedding_dim = 480
elif model_name == "esm2_t33_650M_UR50D":
    embedding_dim = 1280
embeddings = np.zeros((num_sequences, padded_seq_len, embedding_dim), dtype=np.float32)

# Compute embeddings in batches
jitted_esm = jax.jit(esm.apply)
batch_size = 64

print("Starting embedding")
for i in tqdm(range(0, num_sequences, batch_size), unit="batches"):
    batch = encodings_ids[i:i+batch_size]
    batch_embeddings = jitted_esm(esm_params, batch)
    embeddings[i:i+batch_size] = np.array(batch_embeddings)

non_zero_pos = np.any(embeddings != 0, axis=(0, 2))
embeddings = embeddings[:, non_zero_pos, :]
padding_mask = padding_mask[:, non_zero_pos]

assert embeddings.shape[0] == num_sequences, "Incoherent number of embeddings produced compared to input sequences"
print(f"Final embedding array shape: {embeddings.shape} (n_sequences x padded_seq_len x embedding_dim)")
np.savez(file=args.out_file, embedding=embeddings, padding_mask=padding_mask, label=caspase_ids)
print(f"Esm model {embedding_dim}d embeddings saved at: {args.out_file}")
