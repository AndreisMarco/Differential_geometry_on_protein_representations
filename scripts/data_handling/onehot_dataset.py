import pandas as pd

import jax
import jax.numpy as jnp
from jax.nn import one_hot

aa2idx = {"A": 0, "C": 1,"D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,"K": 8, "L": 9, "M": 10, 
          "N": 11, "P": 12,"Q": 13,"R": 14, "S": 15, "T": 16,"V": 17, "W": 18, "Y": 19, "-": 20}
padding_token = list(aa2idx.values())[-1]

idx2aa = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", 
          "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "-"]

def int2onehot(int_seq: jnp.array, flatten=False):
    onehot_seq = one_hot(x=int_seq, num_classes=21)
    if flatten:
        onehot_seq = onehot_seq.reshape(int_seq.shape[0], -1)
    return onehot_seq

def onehot2int(onehot_seq: jnp.array):
    int_seq = jnp.argmax(onehot_seq, axis=-1)
    return int_seq

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encode proteins using esmjax")
    parser.add_argument("-input_file", type=str, default="0_reset/data/caspase_info_filtered.csv", help="Path to the file containing the sequences")
    parser.add_argument("-out_file", type=str, default="0_reset/data/int_representation.npz", help="Output file where to save the embeddings")
    args = parser.parse_args()

    # Load df containing sequences
    df = pd.read_csv(args.input_file, )
    max_len = max(df["len"])

    # Extract and encode sequences
    print(f"Padding to {max_len=} using character {padding_token} as token")
    sequences = df["sequence"].tolist()
    encoded = [[aa2idx[aa] for aa in seq] for seq in sequences]
    padded_encoded = jnp.array([seq + [padding_token] * (max_len - len(seq)) for seq in encoded])
    padding_mask = jnp.array([[1] * len(seq) + [0] *  (max_len - len(seq)) for seq in encoded])
    print(f"Final embedding array shape: {padded_encoded.shape} (n_sequences x padded_seq_len x embedding_dim)")
    
    # Saved encoded sequences, padding mask and respective labels (caspase number)
    labels = df["caspase"].to_numpy()
    jnp.savez(file=args.out_file, embedding=padded_encoded, padding_mask=padding_mask, label=labels)
    print(f"Integer representation saved at {args.out_file}")

    example = padded_encoded[:5]
    onehot_seq = int2onehot(example)
    assert onehot_seq.shape == (5, max_len, 21), \
        f"The int2onehot function returned wrong shape"
    int_seq = onehot2int(onehot_seq)
    assert int_seq.shape == (5, max_len), \
        f"The onehot2int function returned wrong shape"