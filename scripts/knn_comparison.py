import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import yaml

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/encoding_comparison/knn_comparison.yaml', help='Path to yaml configuration file (default: %(default)s)')
    args = parser.parse_args()

    # Read configuration file
    cfg_path = args.cfg
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    def split_path(path, split):
        return os.path.join(path, f"{split}_latents.npz")
    data = {
        path: {
        "train": np.load(split_path(path, "train")),
        "test":  np.load(split_path(path, "test"))
        }
        for path in cfg["latent_paths"]}


    # Safety check tthat the test labels are identical
    test_labels = [data[path]["test"]["label"] for path in cfg["latent_paths"]]    
    first, *rest = test_labels
    for i, arr in enumerate(rest):
            assert np.array_equal(first, arr), f"!Test labels differ in file {i+1}"

    ks = cfg["k"]
    results = {}
    if not os.path.exists(cfg["out_path"]):
        os.makedirs(cfg["out_path"])
    summary_file = os.path.join(cfg["out_path"], "knn_comparison.txt")
    with open(summary_file, 'w') as f:
        for path in cfg["latent_paths"]:
            label = path.split("/")[-1]
            print(label)
            f.write(f"Source of embeddings:\n")
            f.write(f"train: {split_path(path, 'train')}\n")
            f.write(f"test: {split_path(path, 'test')}\n")

            accuracies = []
            for k in ks:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(data[path]["train"]["embedding"], data[path]["train"]["label"])
                preds = knn.predict(data[path]["test"]["embedding"])
                acc = (preds == data[path]["test"]["label"]).mean()
                accuracies.append(acc)
                print(f"{k=:<5}{acc=:.5f}")
                f.write(f"{k=:<5}{acc=:.5f}\n")
            print("\n")
            f.write(f"\n")
            results[label] = accuracies
    
    # Plot comparison
    plt.figure()
    for label, accs in results.items():
        plt.plot(cfg["k"], accs, marker="o", label=label)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xticks(cfg["k"])
    plt.ylim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(cfg["out_path"], "knn_comparison.png")
    plt.savefig(plot_path)
