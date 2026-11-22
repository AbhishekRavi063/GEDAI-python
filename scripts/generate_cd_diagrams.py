import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Directory setup
RESULTS_DIR = 'results'
OUTPUT_DIR = 'results/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
datasets = {
    'BNCI2014_001': 'threshold_sweep_BNCI2014_001.csv',
    'Shin2017A': 'threshold_sweep_Shin2017A.csv'
}

def analyze_epoch_thresholds():
    all_data = []
    
    # Load epoch threshold sweeps
    for ds_name, fname in datasets.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Add dataset column if missing
            if 'dataset' not in df.columns:
                df['dataset'] = ds_name
            # For threshold sweep, enova_threshold=1.0 means reconstruct-all
            all_data.append(df)
            
    if not all_data:
        print("No threshold sweep data found.")
        return
        
    df = pd.concat(all_data, ignore_index=True)
    
    # We want to create a long format dataframe for Friedman/Nemenyi
    # Columns needed: subject_dataset, threshold, metric
    
    # Create unique subject IDs across datasets
    df['sub_id'] = df['dataset'] + "_s" + df['subject'].astype(str)
    
    # Select key thresholds to avoid cluttering the CD diagram
    # e.g., 0.1, 0.5, 0.8, 0.9 (default), 0.95, 1.0 (reconstruct-all)
    target_thresholds = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    
    # Clean up floating point representation issues
    df['enova_threshold'] = df['enova_threshold'].round(2)
    df_filtered = df[df['enova_threshold'].isin(target_thresholds)].copy()
    
    # Format thresholds as strings for plotting
    df_filtered['threshold_name'] = df_filtered['enova_threshold'].apply(
        lambda x: "Reconstruct-All" if x == 1.0 else f"Reject T={x}"
    )
    
    # --- 1. Balanced Accuracy (BA) CD Diagram ---
    metric_ba = 'ba_reject_keep'
    # Pivot to wide format: rows=subjects, cols=thresholds
    ba_pivot = df_filtered.pivot(index='sub_id', columns='threshold_name', values=metric_ba)
    ba_pivot = ba_pivot.dropna()
    
    if len(ba_pivot) > 0:
        # Run Friedman
        stat, p_val = stats.friedmanchisquare(*[ba_pivot[col] for col in ba_pivot.columns])
        print(f"Friedman test for BA: p={p_val:.2e}")
        
        # Post-hoc Nemenyi
        nemenyi_ba = sp.posthoc_nemenyi_friedman(ba_pivot.values)
        nemenyi_ba.columns = ba_pivot.columns
        nemenyi_ba.index = ba_pivot.columns
        
        # Draw CD diagram
        avg_ranks_ba = ba_pivot.rank(axis=1, ascending=False).mean()
        
        plt.figure(figsize=(10, 4))
        # Note: scikit-posthocs >= 0.14 uses sign_plot or critical_difference_diagram
        try:
            sp.critical_difference_diagram(avg_ranks_ba, nemenyi_ba)
            plt.title('Critical Difference Diagram: Balanced Accuracy Across Epoch Rejection Thresholds')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'cd_diagram_ba.pdf'), bbox_inches='tight')
            print("Saved cd_diagram_ba.pdf")
        except AttributeError:
            print("critical_difference_diagram not found in scikit_posthocs")
        plt.close()
    
    # --- 2. Information Transmission Rate (ITR) CD Diagram ---
    metric_itr = 'itr_effective_reject'
    if metric_itr in df_filtered.columns:
        itr_pivot = df_filtered.pivot(index='sub_id', columns='threshold_name', values=metric_itr)
        itr_pivot = itr_pivot.dropna()
        
        if len(itr_pivot) > 0:
            stat, p_val = stats.friedmanchisquare(*[itr_pivot[col] for col in itr_pivot.columns])
            print(f"Friedman test for ITR: p={p_val:.2e}")
            
            nemenyi_itr = sp.posthoc_nemenyi_friedman(itr_pivot.values)
            nemenyi_itr.columns = itr_pivot.columns
            nemenyi_itr.index = itr_pivot.columns
            
            avg_ranks_itr = itr_pivot.rank(axis=1, ascending=False).mean()
            
            plt.figure(figsize=(10, 4))
            try:
                sp.critical_difference_diagram(avg_ranks_itr, nemenyi_itr)
                plt.title('Critical Difference Diagram: ITR (Bits/min) Across Epoch Rejection Thresholds')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'cd_diagram_itr.pdf'), bbox_inches='tight')
                print("Saved cd_diagram_itr.pdf")
            except AttributeError:
                pass
            plt.close()

    # --- 3. Generate Statistical Table (P-values and Cohen's D) ---
    # We will compare Reconstruct-All vs Reject T=0.90 (Default) vs Reject T=0.50
    compare_cols = ['Reconstruct-All', 'Reject T=0.9', 'Reject T=0.5']
    if all(c in ba_pivot.columns for c in compare_cols):
        res_matrix = pd.DataFrame(index=compare_cols, columns=compare_cols, dtype=object)
        
        for i, c1 in enumerate(compare_cols):
            for j, c2 in enumerate(compare_cols):
                if i == j:
                    res_matrix.loc[c1, c2] = "-"
                elif i < j: # Upper triangle: Cohen's d
                    diff = ba_pivot[c1] - ba_pivot[c2]
                    d = diff.mean() / diff.std(ddof=1) if diff.std() > 0 else 0
                    res_matrix.loc[c1, c2] = f"{d:.2f}"
                else: # Lower triangle: Wilcoxon p-value
                    try:
                        _, p = stats.wilcoxon(ba_pivot[c1], ba_pivot[c2])
                        res_matrix.loc[c1, c2] = f"{p:.1e}"
                    except Exception as e:
                        res_matrix.loc[c1, c2] = "N/A"
                        
        res_matrix.to_csv(os.path.join(OUTPUT_DIR, 'stats_matrix_ba.csv'))
        print("Generated stats_matrix_ba.csv")

if __name__ == '__main__':
    analyze_epoch_thresholds()
    print("Done generating diagrams.")
