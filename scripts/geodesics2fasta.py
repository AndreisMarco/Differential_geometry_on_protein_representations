import os
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract proteins from geodesic endpoints to FASTA format')
    parser.add_argument('--geodesics_file', type=str, default='logs/var_adj_VAE/esm_var_adj_1280d_kl_5e-3_5e-4/geodesics/geodesics.npz',help='Path to geodesics.npz file')
    parser.add_argument('--csv', type=str, default='data/caspase_info_filtered.csv', help='Path to caspase info CSV file')
    parser.add_argument('--out_file', type=str, default='logs/var_adj_VAE/esm_var_adj_1280d_kl_5e-3_5e-4/evolutionary_analysis/geodesic_proteins.fasta', help='Output FASTA file path')
    args = parser.parse_args()
    
    # Load geodesics idxs
    print(f"Loading geodesics from {args.geodesics_file}")
    data = np.load(args.geodesics_file)
    idx_pairs = data['idx']  # Shape: (num_geodesics, 2)
    # Get unique idxs
    unique_idxs = np.unique(idx_pairs.flatten())
    print(f"Found {len(unique_idxs)} unique proteins from {len(idx_pairs)} geodesics")

    # Load protein information
    print(f"Loading protein data from {args.csv}")
    df = pd.read_csv(args.csv)

    # Create FASTA file
    print(f"Writing FASTA to {args.out_file}")
    with open(args.out_file, 'w') as f:
        for idx in unique_idxs:
            protein_info = df.iloc[idx]
            # Create header 
            header = f">{protein_info['id']}|caspase_{protein_info['caspase']}|{protein_info['species']}"
            # Write header and sequence
            f.write(header + '\n')
            f.write(protein_info['sequence'] + '\n')
    
    print(f"Successfully wrote {len(unique_idxs)} proteins to {args.out_file}")
