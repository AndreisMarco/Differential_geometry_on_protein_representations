import os
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse

def extract_patristic_distances(tree_file):
    """Extract pairwise patristic distances from phylogenetic tree"""
    # Load tree
    tree = Phylo.read(tree_file, "newick")
    # Get all leaf nodes
    terminals = tree.get_terminals()
    n_terminals = len(terminals)
    print(f"Found {n_terminals} sequences in tree")
    
    # Create distance matrix
    distance_matrix = np.zeros((n_terminals, n_terminals))
    protein_names = [term.name for term in terminals]
    
    # Calculate patristic distances (sum of branch lengths)
    print("Computing pairwise distances")
    for i, term1 in enumerate(terminals):
        for j, term2 in enumerate(terminals):
            if i < j:
                dist = tree.distance(term1, term2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    return distance_matrix, protein_names

def load_geodesic_data(geodesics_file, csv_file):
    """Load geodesic data and protein information"""
    print(f"\nLoading geodesic data from {geodesics_file}...")
    data = np.load(geodesics_file)
    idx_pairs = data['idx']
    geodesic_lengths = data['length']
    euclidean_dists = data['euclidean_dist']
    
    print(f"Found {len(idx_pairs)} geodesic pairs")
    
    # Load protein information
    print(f"Loading protein info from {csv_file}")
    df_proteins = pd.read_csv(csv_file)
    
    return idx_pairs, geodesic_lengths, euclidean_dists, df_proteins

def create_comparison_dataframe(evolutionary_dist_matrix, protein_names, 
                                idx_pairs, geodesic_lengths, euclidean_dists, 
                                df_proteins):
    """Match geodesic pairs with evolutionary distances"""
    print("\nMatching geodesic pairs with evolutionary distances")
    
    # Create mapping from protein ID to position in tree
    name_to_pos = {}
    for pos, name in enumerate(protein_names):
        # Extract protein ID (before first |)
        protein_id = name.split('|')[0] if '|' in name else name
        name_to_pos[protein_id] = pos
    
    # Build comparison list
    comparisons = []
    skipped = 0
    
    for i, (idx1, idx2) in enumerate(idx_pairs):
        protein_id1 = df_proteins.iloc[idx1]['id']
        protein_id2 = df_proteins.iloc[idx2]['id']
        
        # Find positions in tree
        if protein_id1 in name_to_pos and protein_id2 in name_to_pos:
            pos1 = name_to_pos[protein_id1]
            pos2 = name_to_pos[protein_id2]
            
            comparisons.append({
                'protein1': protein_id1,
                'protein2': protein_id2,
                'idx1': idx1,
                'idx2': idx2,
                'evolutionary_distance': evolutionary_dist_matrix[pos1, pos2],
                'geodesic_length': geodesic_lengths[i],
                'euclidean_distance': euclidean_dists[i]
            })
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} pairs (proteins not found in tree)")
    
    df = pd.DataFrame(comparisons)
    print(f"Successfully matched {len(df)} pairs")
    
    # In create_comparison_dataframe, add this:
    print(f"\nTree name examples:")
    for i in range(min(5, len(protein_names))):
        print(f"  {protein_names[i]}")

    print(f"\nCSV ID examples:")
    for i in range(min(5, len(df_proteins))):
        print(f"  {df_proteins.iloc[i]['id']}")

    print(f"\nExtracted IDs from tree:")
    for name in protein_names[:5]:
        extracted = name.split('|')[0] if '|' in name else name
        print(f"  {name} → {extracted}")

    return df

def compute_correlations(df):
    """Compute and display Pearson correlations"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Geodesic vs Evolutionary
    corr_geo, p_geo = pearsonr(df['geodesic_length'], df['evolutionary_distance'])
    
    # Euclidean vs Evolutionary
    corr_euc, p_euc = pearsonr(df['euclidean_distance'], df['evolutionary_distance'])
    
    print(f"\nGeodesic Length vs Evolutionary Distance:")
    print(f"  Pearson r = {corr_geo:.4f}")
    print(f"  p-value   = {p_geo:.4e}")
    
    print(f"\nEuclidean Distance vs Evolutionary Distance:")
    print(f"  Pearson r = {corr_euc:.4f}")
    print(f"  p-value   = {p_euc:.4e}")
    
    improvement = corr_geo - corr_euc
    print(f"\n{'✓' if improvement > 0 else '✗'} Improvement: {improvement:+.4f}")
    
    if improvement > 0:
        pct_improvement = (improvement / abs(corr_euc)) * 100
        print(f"  ({pct_improvement:+.1f}% relative improvement)")
    
    print("="*70 + "\n")
    
    return {
        'geodesic_r': corr_geo,
        'geodesic_p': p_geo,
        'euclidean_r': corr_euc,
        'euclidean_p': p_euc,
        'improvement': improvement
    }

def plot_comparison(df, correlations, output_file):
    """Create scatter plots comparing distances"""
    print(f"Creating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Geodesic vs Evolutionary
    ax = axes[0]
    ax.scatter(df['geodesic_length'], df['evolutionary_distance'],
              alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Geodesic Length', fontsize=13)
    ax.set_ylabel('Evolutionary Distance (Patristic)', fontsize=13)
    ax.set_title(f"Geodesic vs Evolutionary\nPearson r = {correlations['geodesic_r']:.4f}", 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Fit line
    z = np.polyfit(df['geodesic_length'], df['evolutionary_distance'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['geodesic_length'].min(), df['geodesic_length'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax.legend()
    
    # Euclidean vs Evolutionary
    ax = axes[1]
    ax.scatter(df['euclidean_distance'], df['evolutionary_distance'],
              alpha=0.6, s=30, color='orange', edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Euclidean Distance', fontsize=13)
    ax.set_ylabel('Evolutionary Distance (Patristic)', fontsize=13)
    ax.set_title(f"Euclidean vs Evolutionary\nPearson r = {correlations['euclidean_r']:.4f}", 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Fit line
    z = np.polyfit(df['euclidean_distance'], df['evolutionary_distance'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['euclidean_distance'].min(), df['euclidean_distance'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Compare evolutionary distances with geodesic and Euclidean distances'
    )
    parser.add_argument('--tree', type=str, default="0_reset/new_logs/var_adj_VAE/esm_var_adj_1280d_kl_1e-3_1e-2/evolutionary_analysis/geodesic_proteins.tree", help='Path to tree file (Newick format)')
    parser.add_argument('--geodesics', type=str, default="0_reset/new_logs/var_adj_VAE/esm_var_adj_1280d_kl_1e-3_1e-2/geodesics/geodesics.npz", help='Path to geodesics.npz file')
    parser.add_argument('--csv', type=str, default="0_reset/data/caspase_info_filtered.csv", help='Path to caspase_info_filtered.csv')
    parser.add_argument('--output-dir', type=str, default='0_reset/new_logs/var_adj_VAE/esm_var_adj_1280d_kl_1e-3_1e-2/evolutionary_analysis/', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Extract evolutionary distances from tree
    dist_matrix, protein_names = extract_patristic_distances(args.tree)
    
    # Step 2: Load geodesic data
    idx_pairs, geo_lengths, euc_dists, df_proteins = load_geodesic_data(
        args.geodesics, args.csv
    )
    
    # Step 3: Create comparison dataframe
    df_comparisons = create_comparison_dataframe(
        dist_matrix, protein_names, idx_pairs, geo_lengths, euc_dists, df_proteins
    )
    
    # Save comparisons
    comparison_file = os.path.join(args.output_dir, 'distance_comparisons.csv')
    df_comparisons.to_csv(comparison_file, index=False)
    print(f"\nComparisons saved to {comparison_file}")
    
    # Step 4: Compute correlations
    correlations = compute_correlations(df_comparisons)
    
    # Step 5: Create plots
    plot_file = os.path.join(args.output_dir, 'distance_correlation_plot.pdf')
    plot_comparison(df_comparisons, correlations, plot_file)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'correlation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Evolutionary Distance vs Latent Space Distance\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of protein pairs analyzed: {len(df_comparisons)}\n\n")
        f.write(f"Geodesic Length vs Evolutionary Distance:\n")
        f.write(f"  Pearson r = {correlations['geodesic_r']:.4f}\n")
        f.write(f"  p-value   = {correlations['geodesic_p']:.4e}\n\n")
        f.write(f"Euclidean Distance vs Evolutionary Distance:\n")
        f.write(f"  Pearson r = {correlations['euclidean_r']:.4f}\n")
        f.write(f"  p-value   = {correlations['euclidean_p']:.4e}\n\n")
        f.write(f"Improvement: {correlations['improvement']:+.4f}\n")
    
    print(f"Summary saved to {summary_file}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()