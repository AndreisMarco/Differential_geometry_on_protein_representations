import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pandas as pd

import jax.numpy as jnp
import flax.nnx as nnx

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from utils import load_VAE, load_esm_head

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encode proteins using esmjax")
    parser.add_argument("-vae_path", type=str, default="logs/esm_1280/" , help="Path to the file the esm encodings")
    parser.add_argument("-esm_model", type=str, default="esm2_t33_650M_UR50D", help="Model from which to take the head to be used for decoding")
    parser.add_argument("-protein_df_file", type=str, default="data/caspase_info_filtered.csv" , help="Path to the file the esm encodings")
    args = parser.parse_args()

    # Create directory for saving the decoded sequences
    model_path = args.vae_path
    decoded_dir = os.path.join(model_path, "decoded_sequences")
    os.makedirs(decoded_dir, exist_ok=True)
    # Load the vae model
    rngs = nnx.Rngs(42)
    model, model_cfg = load_VAE(rngs=rngs, model_path=model_path)


    # Load the esm model head
    esm_head = load_esm_head(model_name=args.esm_model)

    # Load geodesics latents and the idx of the ends
    geodesics_file = os.path.join(model_path, "geodesics", "geodesics.npz")
    data = jnp.load(geodesics_file)
    idx = data["idx"]
    geodesics = data["geodesics"]

    # Only decode midpoints
    midpoint = geodesics.shape[1] // 2
    midpoint_latents = geodesics[:,midpoint,:]
    mu, std = model.decoder(midpoint_latents)

    # Decode embeddings into protein sequences
    logits = esm_head(mu)  # shape: (num_geodesics, seq_len, num_tokens)

    # Get the most likely token at each position
    token_ids = jnp.argmax(logits, axis=-1)

    # Convert token ids to sequences
    esm_alphabet = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 
                    'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 
                    'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 
                    'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 
                    'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 
                    'B': 25, 'U': 26, 'Z': 27, 'O': 28, 
                    '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
    id_to_aa = {idx: token for token, idx in esm_alphabet.items()}
    sequences = ["ALRKB", "ERDLP", "FTKID"]
    # for seq in token_ids:
    #     seq_str = "".join([esm_alphabet[int(t)] for t in seq])
    #     sequences.append(seq_str)

    # Load df with the informations about the endpoints
    df = pd.read_csv(args.protein_df_file)

    # Save fasta files containing the decoded midpoint seq
    # and its start and end 
    for i in range(len(sequences)):
        start_record = SeqRecord(
            id=df.iloc[idx[i,0]]["id"],
            description=f"Caspase {df.iloc[idx[i,0]]['caspase']}",
            seq=Seq(df.iloc[idx[i,0]]["sequence"]))
        
        end_record = SeqRecord(
            id=df.iloc[idx[i,1]]["id"],
            description=f"Caspase {df.iloc[idx[i,1]]['caspase']}",
            seq=Seq(df.iloc[idx[i,1]]["sequence"]))

        mid_record = SeqRecord(
            id="MiddlePoint",
            description=f"Decoded from {args.esm_model}",
            seq=Seq(sequences[i]))
        
        records = [start_record, mid_record, end_record]
        fasta_file = os.path.join(decoded_dir, f"{i}.fasta")
        SeqIO.write(records, fasta_file, "fasta")


