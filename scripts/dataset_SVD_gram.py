import numpy as np
import matplotlib.pyplot as plt
from data_handling.onehot_dataset import int2onehot

def load_dataset(file, processing_fn):
    data = np.load(file)
    embeddings = data["embedding"]
    labels = data["label"]
    return processing_fn(embeddings, labels)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encode proteins using esmjax")
    parser.add_argument("-input_file", type=str, default="data/datasets/esm_representation_1280d.npz", help="Path to the file containing the sequences")
    parser.add_argument("-out_file", type=str, default="data/variance_explained_1280d.pdf", help="Path to save the plot to")
    parser.add_argument( "-type", type=str, default="esm_full", help="Type of dataset")
    args = parser.parse_args()
    
    # Select preprocessing
    if args.type == "onehot":
        processing_fn = lambda X, labels: int2onehot(X).reshape(X.shape[0], -1)
    if args.type == "esm_avg":
        processing_fn = lambda X, labels: np.mean(X, axis=1)
    if args.type == "esm_full":
        processing_fn = lambda X, labels: X.reshape(X.shape[0], -1)
    if args.type == "esm_filter_full":
        def processing_fn(X, labels):
            mask = labels == 3
            X = X[mask]
            X = X.reshape(X.shape[0], -1)
            return X

    # Load data
    X = load_dataset(file=args.input_file, processing_fn=processing_fn)

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # Compute the Gram matrix 
    G = X_centered @ X_centered.T

    # Eigen-decomposition 
    eigvals, eigvecs = np.linalg.eigh(G)
    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute singular values
    S = np.sqrt(eigvals.clip(min=0))  # clip to avoid small negative numerical errors

    # Explained variance
    explained_variance_ratio = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Components for 90% and 95% variance
    k_90 = np.searchsorted(cumulative_variance, 0.90) + 1
    k_95 = np.searchsorted(cumulative_variance, 0.95) + 1

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    x_vals = np.arange(1, len(cumulative_variance) + 1)
    plt.plot(x_vals, cumulative_variance, label="Cumulative explained variance")
    plt.axhline(0.90, color='r', linestyle='--',
                label=f"90% variance threshold ({k_90} features)")
    plt.axhline(0.95, color='g', linestyle='--',
                label=f"95% variance threshold ({k_95} features)")
    plt.axvline(k_90, color='r', linestyle=':')
    plt.axvline(k_95, color='g', linestyle=':')

    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("SVD Explained Variance (Gram Matrix method)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_file, format="pdf")
