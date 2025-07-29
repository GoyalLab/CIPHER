import os
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import pickle
from tqdm import tqdm

from src.preprocess import get_data


def compute_covariance(X):
    return np.cov(X, rowvar=False)

def compute_average_response(X0, X1):
    return X1.mean(axis=0) - X0.mean(axis=0)

def compute_soft_fraction(Sigma, u, threshold_mode='fraction_variance', threshold_value=0.7):
    lambda_vals, V = np.linalg.eigh(Sigma)
    idx = np.argsort(lambda_vals)[::-1]
    lambda_vals = lambda_vals[idx]
    V = V[:, idx]
    c = V.T @ u
    c2 = c ** 2
    if threshold_mode == 'fraction_variance':
        total_var = np.sum(lambda_vals)
        cum_var = np.cumsum(lambda_vals)
        soft_indices = np.where(cum_var <= threshold_value * total_var)[0]
    elif threshold_mode == 'relative_max':
        lambda_max = np.max(lambda_vals)
        soft_indices = np.where(lambda_vals >= threshold_value * lambda_max)[0]
    elif threshold_mode == 'elbow':
        diffs = np.diff(lambda_vals)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs)
        soft_indices = np.arange(elbow_idx + 1)
    else:
        raise ValueError("Invalid threshold_mode.")
    f_soft = np.sum(c2[soft_indices]) / np.sum(c2)
    return f_soft, soft_indices

def r2_score_standart(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate R2 coefficient for y (truth) vs. y^ (prediction)"""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    valid = np.abs(y) > 0
    if not np.any(valid):
        return np.nan
    return 1 - ss_res[valid] / (ss_tot[valid] + eps)

def r2_score(y: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate R2 coefficient for x (truth) vs. y^ (prediction)"""
    valid = np.abs(x) > 0
    if not np.any(valid):
        return np.nan
    ss_res = np.sqrt(np.sum((y[valid] - x[valid]) ** 2))
    ss_tot = np.sqrt(np.sum(x[valid] ** 2))
    return 1 - ss_res / (ss_tot + eps)

def run(data_path, save_dir="r2_histograms"):
    # Read adata to file
    adata, X0, _ = get_data(0, data_path, qc=True, save=False)
    gene_names = np.array(adata.var_names.tolist())
    X0_dense = X0.toarray() if issparse(X0) else X0
    # Compute average
    X0_mean = X0.mean(axis=0)

    Sigma_real = compute_covariance(X0_dense)

    # Mean-field null model: shuffle each gene across cells
    X0_gene_shuffled = X0_dense.copy()
    for g in range(X0_gene_shuffled.shape[1]):
        np.random.shuffle(X0_gene_shuffled[:, g])
    Sigma_null = compute_covariance(X0_gene_shuffled)

    # Full shuffle: shuffle all values of X0 (cells × genes), then reshape
    X0_flat = np.array(X0_dense).flatten()  # fully detached copy
    X0_shuffled_flat = np.random.permutation(X0_flat)  # avoids in-place modification
    X0_full_shuffled = X0_shuffled_flat.reshape(X0_dense.shape)
    Sigma_rand = compute_covariance(X0_full_shuffled)
 
    perturbations = [p for p in adata.obs['perturbation'].unique() if p != 'control']

    R2_real, R2_null, R2_rand = [], [], []
    f_soft_scores = []
    pert_names = []
    # Pre-compute arrays and avoid repeated computations
    valid_perts = [p for p in perturbations if p in gene_names]
    gene_indices = np.array([np.where(gene_names == p)[0][0] for p in valid_perts])
    
    # Vectorize perturbation response calculation
    X1_means = np.vstack([
        adata[adata.obs['perturbation'] == p].X.mean(axis=0).A1 
        if issparse(adata[adata.obs['perturbation'] == p].X) 
        else adata[adata.obs['perturbation'] == p].X.mean(axis=0) 
        for p in valid_perts
    ])
    delta_X_matrix = (X1_means - X0_mean).T
    # Calculate valid deltas
    valid_mask = (np.abs(delta_X_matrix) > 0).T * 1
    delta_X_matrix *= valid_mask

    # Vectorized R2 calculation
    def predict_r2_batch(Sigma, gene_indices, delta_X_matrix, eps=1e-8):
        sigma_cols = Sigma[:, gene_indices]
        u_opts = np.sum(sigma_cols @ delta_X_matrix.T, axis=1) / (
            np.sum(sigma_cols @ sigma_cols, axis=0) + eps
        )
        preds = u_opts[:, None] * sigma_cols
        preds *= valid_mask
        r2 = 1 - np.sum((delta_X_matrix - preds)**2, axis=0) / (np.sum(delta_X_matrix**2, axis=0)+1e-8)
        return r2

    # Calculate all R2 scores at once
    R2_real.extend(predict_r2_batch(Sigma_real, gene_indices, delta_X_matrix))
    R2_null.extend(predict_r2_batch(Sigma_null, gene_indices, delta_X_matrix))
    R2_rand.extend(predict_r2_batch(Sigma_rand, gene_indices, delta_X_matrix))
    
    # Calculate f_soft scores
    f_soft_scores.extend([
        compute_soft_fraction(Sigma_real, Sigma_real[:, idx])[0] 
        for idx in gene_indices
    ])
    pert_names.extend(valid_perts)
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(data_path).replace('.h5ad', '').replace('.pkl', '')

    # R2 histogram
    plt.figure(figsize=(8, 6))
    plt.hist(R2_real, bins=30, alpha=0.7, label='Real Σ', density=True)
    plt.hist(R2_null, bins=30, alpha=0.7, label='Shuffled X₀', density=True)
    plt.hist(R2_rand, bins=30, alpha=0.7, label='Fully Shuffled X₀', density=True)
    plt.xlabel("R² Score", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.yscale('log')
    plt.title(f"R² Distributions: {base_name}", fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_r2_histogram.svg"))
    plt.close()

    # f_soft histogram
    plt.figure(figsize=(8, 6))
    plt.hist(f_soft_scores, bins=30, alpha=0.8, density=True)
    plt.xlabel("f_soft", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.title(f"f_soft Distribution: {base_name}", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_fsoft_histogram.svg"))
    plt.close()

    df = pd.DataFrame({
        "perturbation": pert_names,
        "R2_real": R2_real,
        "R2_null": R2_null,
        "R2_rand": R2_rand,
        "f_soft": f_soft_scores
    })
    df.to_csv(os.path.join(save_dir, f"{base_name}_results.csv"), index=False)

    print(f"Completed {base_name}")
    return df

def full_analysis_with_nulls_soft_and_plots(data_path, save_dir="r2_histograms"):
    # Read adata to file
    adata, X0, _ = get_data(0, data_path, qc=True, save=False)
    gene_names = np.array(adata.var_names.tolist())
    X0_dense = X0.toarray() if issparse(X0) else X0
    # Compute average
    X0_mean = X0.mean(axis=0)

    Sigma_real = compute_covariance(X0_dense)

    # Mean-field null model: shuffle each gene across cells
    X0_gene_shuffled = X0_dense.copy()
    for g in range(X0_gene_shuffled.shape[1]):
        np.random.shuffle(X0_gene_shuffled[:, g])
    Sigma_null = compute_covariance(X0_gene_shuffled)

    # Full shuffle: shuffle all values of X0 (cells × genes), then reshape
    X0_flat = np.array(X0_dense).flatten()  # fully detached copy
    X0_shuffled_flat = np.random.permutation(X0_flat)  # avoids in-place modification
    X0_full_shuffled = X0_shuffled_flat.reshape(X0_dense.shape)
    Sigma_rand = compute_covariance(X0_full_shuffled)
 
    perturbations = [p for p in adata.obs['perturbation'].unique() if p != 'control']

    R2_real, R2_null, R2_rand = [], [], []
    f_soft_scores = []
    pert_names = []

    for pert in tqdm(perturbations[:10], desc="Calculating R2s"):
        # Skip perturbations that are not in features of adata
        if pert not in gene_names:
            continue
        # Subset perturbation data
        gene_idx = np.where(gene_names == pert)[0][0]
        X1 = adata[adata.obs['perturbation'] == pert].X
        X1 = X1.toarray() if issparse(X1) else X1
        # Get average perturbation response to control
        delta_X = X1.mean(axis=0) - X0_mean

        def predict_r2(Sigma, eps: float = 1e-8):
            sigma_col = Sigma[:, gene_idx]
            u_opt = np.dot(sigma_col, delta_X) / (np.dot(sigma_col, sigma_col) + eps)
            pred = u_opt * sigma_col
            return r2_score(sigma_col, pred, eps=eps)
        
        # Predict R2 for every Sigma
        R2_real.append(predict_r2(Sigma_real))
        R2_null.append(predict_r2(Sigma_null))
        R2_rand.append(predict_r2(Sigma_rand))
        f_soft, _ = compute_soft_fraction(Sigma_real, Sigma_real[:, gene_idx])
        f_soft_scores.append(f_soft)
        pert_names.append(pert)

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(data_path).replace('.h5ad', '').replace('.pkl', '')

    # R2 histogram
    plt.figure(figsize=(8, 6))
    plt.hist(R2_real, bins=30, alpha=0.7, label='Real Σ', density=True)
    plt.hist(R2_null, bins=30, alpha=0.7, label='Shuffled X₀', density=True)
    plt.hist(R2_rand, bins=30, alpha=0.7, label='Fully Shuffled X₀', density=True)
    plt.xlabel("R² Score", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.yscale('log')
    plt.title(f"R² Distributions: {base_name}", fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_r2_histogram.svg"))
    plt.close()

    # f_soft histogram
    plt.figure(figsize=(8, 6))
    plt.hist(f_soft_scores, bins=30, alpha=0.8, density=True)
    plt.xlabel("f_soft", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.title(f"f_soft Distribution: {base_name}", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_fsoft_histogram.svg"))
    plt.close()

    df = pd.DataFrame({
        "perturbation": pert_names,
        "R2_real": R2_real,
        "R2_null": R2_null,
        "R2_rand": R2_rand,
        "f_soft": f_soft_scores
    })
    df.to_csv(os.path.join(save_dir, f"{base_name}_results.csv"), index=False)

    print(f"Completed {base_name}")
    return df

def plot_r2_comparison(datasets, label, csv_dir, save_dir, kind="null"):
    r2_real_all = []
    r2_comp_all = []

    for name in datasets:
        csv_path = os.path.join(csv_dir, f"{name}_results.csv")
        if not os.path.exists(csv_path):
            print(f"Missing CSV for {name}, skipping")
            continue
        df = pd.read_csv(csv_path)
        if "R2_real" not in df or f"R2_{kind}" not in df:
            continue
        r2_real_all.extend(df["R2_real"].dropna().values)
        r2_comp_all.extend(df[f"R2_{kind}"].dropna().values)

    if not r2_real_all or not r2_comp_all:
        print(f"No data for {label} ({kind})")
        return

    # Perform KS test
    ks_stat, p_value = ks_2samp(r2_real_all, r2_comp_all)
    print(f"KS test for {label} ({kind}): stat = {ks_stat:.3f}, p = {p_value:.3e}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(r2_real_all, bins=30, alpha=1, density=True, label="Real Σ", color="#ffa556")
    plt.hist(r2_comp_all, bins=30, alpha=1, density=True, label=f"{'Shuffled X₀' if kind == 'null' else 'Fully Shuffled X₀'}", color='#a58a72')
    plt.xlabel("R² Score", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.yscale("log")
    plt.xlim(0, 1)
    plt.ylim(1e-2, 1e2)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([1e-2, 1e-1, 1, 10, 100])
    plt.title(f"{label}: R² Real vs {kind.capitalize()}", fontsize=20)
    plt.text(0.05, 80, f"KS = {ks_stat:.3f}\nP = {p_value:.2e}", fontsize=13, bbox=dict(facecolor='white', alpha=0.7))
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{label}_R2_vs_{kind}.svg"))
    plt.close()
    print(f"Saved: {label}_R2_vs_{kind}.svg")

def compute_predicted_response(Sigma, delta_X, gene_idx, epsilon=1e-8):
    #sigma_col = Sigma[:, gene_idx]
    sigma_col = Sigma['PIP'][gene_idx]
    u_opt = np.dot(sigma_col, delta_X) / (np.dot(sigma_col, sigma_col) + epsilon)
    pred = u_opt * sigma_col
    return pred, u_opt

def plot_best_r2_perturbation(dataset_names, data_dir="data", cov_dir="covariance_outputs", r2_dir="r2_error_analysis", plot_dir="dx_plots"):
    os.makedirs(plot_dir, exist_ok=True)

    for base_name in dataset_names:
        r2_path = os.path.join(r2_dir, f"{base_name}_results.csv")
        if not os.path.exists(r2_path):
            print(f"R² CSV not found for {base_name}, skipping")
            continue

        df = pd.read_csv(r2_path)
        if df.empty or "perturbation" not in df or "R2_real" not in df:
            print(f"Invalid or missing data for {base_name}, skipping")
            continue

        best_row = df.loc[df["R2_real"].idxmax()]
        best_pert = best_row["perturbation"]

        print(f"{base_name}: highest R² = {best_row['R2_real']:.3f} for {best_pert}")

        # Load sigma
        with open(os.path.join(cov_dir, f"{base_name}_usamples_with_lr.pkl"), "rb") as s:
            Sigma = pickle.load(s)
        # Load adata
        adata_p = os.path.join(data_dir, f"{base_name}.h5ad")
        adata, X0, _ = get_data(0, adata_p)
        
        gene_names = np.array(adata.var_names)

        if best_pert not in gene_names:
            print(f"Perturbation {best_pert} not in gene names, skipping")
            continue

        gene_idx = np.where(gene_names == best_pert)[0][0]
        X_control = adata[adata.obs["perturbation"] == "control"].X
        X_pert = adata[adata.obs["perturbation"] == best_pert].X
        X_control = X_control.toarray() if issparse(X_control) else X_control
        X_pert = X_pert.toarray() if issparse(X_pert) else X_pert

        delta_X = compute_average_response(X_control, X_pert)
        pred, u_opt = compute_predicted_response(Sigma, delta_X, gene_idx)

        # === Styled plot with blue line (ΔX true) and red dots (prediction) ===
        plt.figure(figsize=(12, 5))
        x = np.arange(len(delta_X))
        
        # Blue line: true ΔX values
        plt.plot(x, delta_X, color="blue", linewidth=1.5, label="Measured ΔX (true)")
        
        # Red dots: predicted ΔX values
        plt.scatter(x, pred, color="red", s=25, alpha=0.9, label="Predicted ΔX from single perturbation")
        
        plt.xlabel("Gene Index", fontsize=14)
        plt.ylabel("ΔX", fontsize=14)
        plt.title(f"Best Perturbation: {best_pert} ({base_name}, R² = {best_row['R2_real']:.3f})", fontsize=16)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_name}_dX_lines_dots.svg"))
        plt.close()



        # === Scatter plot ===
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(delta_X, pred, alpha=0.7, edgecolor='k')
        plt.plot([delta_X.min(), delta_X.max()], [delta_X.min(), delta_X.max()], 'k--', lw=1.5, label=r"$y = x$")
        plt.xlabel(r"$\Delta X$ (observed)", fontsize=14)
        plt.ylabel(r"$\Sigma u$ (predicted)", fontsize=14)
        plt.title(f"{base_name}: {best_pert}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_name}_dX_true_pred.svg"))
        plt.close()
        print(f"Plots saved for {base_name} ({best_pert})")

def plot_r2_histograms_for_dataset(dataset_name, csv_dir, save_dir):
    file_path = os.path.join(csv_dir, f"{dataset_name}_results.csv")
    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        return

    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["R2_real", "R2_null", "R2_rand"]):
        print(f"Missing required columns in {dataset_name}")
        return

    real = df["R2_real"].dropna().values
    meanfield = df["R2_null"].dropna().values
    shuffled = df["R2_rand"].dropna().values

    # KS Tests
    ks_mf, p_mf = ks_2samp(real, meanfield)
    ks_shuff, p_shuff = ks_2samp(real, shuffled)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(real, bins=30, density=True, alpha=0.6, label="Real Σ")
    plt.hist(shuffled, bins=30, density=True, alpha=0.6, label=f"Shuffled Σ (KS={ks_shuff:.2g}, p={p_shuff:.1e})")
    plt.hist(meanfield, bins=30, density=True, alpha=0.6, label=f"Meanfield Σ (KS={ks_mf:.2g}, p={p_mf:.1e})")

    plt.xlabel("R² Score", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(f"{dataset_name}: R² Distributions", fontsize=18)
    plt.yscale("log")
    plt.xlim(0, 1)
    plt.ylim(1e-2, 1e2)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([1e-2, 1e-1, 1, 10, 100])
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{dataset_name}_r2_all_hist.svg")
    plt.savefig(save_path, format="svg")
    plt.close()
    print(f"Saved: {save_path}")

def permutation_test(x, y, n_permutations=100000, seed=42):
    np.random.seed(seed)
    diffs = x - y
    observed = np.mean(diffs)
    signs = np.random.choice([-1, 1], size=(n_permutations, len(diffs)))
    permuted_means = np.mean(signs * diffs, axis=1)
    p = np.mean(np.abs(permuted_means) >= np.abs(observed))
    return observed, p

def plot_r2_dataset_means_grouped(dataset_names, label, csv_dir, save_dir):
    r2_real_all, r2_null_all, r2_shuff_all = [], [], []

    dataset_means = []
    valid_names = []

    for base_name in dataset_names:
        file = f"{base_name}_results.csv"
        path = os.path.join(csv_dir, file)
        if not os.path.exists(path):
            print(f"Missing file: {file}, skipping")
            continue

        df = pd.read_csv(path)
        if "R2_real" not in df or "R2_null" not in df or "R2_rand" not in df:
            continue

        r2_real = df["R2_real"].dropna().values
        r2_null = df["R2_null"].dropna().values
        r2_shuff = df["R2_rand"].dropna().values
        n = min(len(r2_real), len(r2_null), len(r2_shuff))

        r2_real = r2_real[:n]
        r2_null = r2_null[:n]
        r2_shuff = r2_shuff[:n]

        r2_real_all.append(np.mean(r2_real))
        r2_null_all.append(np.mean(r2_null))
        r2_shuff_all.append(np.mean(r2_shuff))

        dataset_means.append([np.mean(r2_shuff), np.mean(r2_null), np.mean(r2_real)])
        valid_names.append(base_name)

    if len(dataset_means) == 0:
        print(f"No valid data for group: {label}")
        return

    dataset_means = np.array(dataset_means)
    r2_shuff = dataset_means[:, 0]
    r2_null = dataset_means[:, 1]
    r2_real = dataset_means[:, 2]

    # Test and plot: meanfield (null) vs real
    w_stat_null, p_null = wilcoxon(r2_real, r2_null, alternative="greater")
    w_stat_shuff, p_shuff = wilcoxon(r2_real, r2_shuff, alternative="greater")
    print(f"\n📊 {label} Wilcoxon one-sided tests:")
    print(f"   Meanfield Σ vs Real: W = {w_stat_null:.3f}, p = {p_null:.3e}")
    print(f"   Shuffled Σ vs Real:  W = {w_stat_shuff:.3f}, p = {p_shuff:.3e}")

    # Plot
    x_pos = [0, 1, 2]
    cmap = cm.get_cmap("tab10", len(dataset_means))
    plt.figure(figsize=(10, 10))

    for i, row in enumerate(dataset_means):
        plt.plot(x_pos, row, color=cmap(i), linewidth=2, alpha=0.9, label=valid_names[i])
        plt.scatter(x_pos, row, color=cmap(i), s=60)

    mean_vals = np.mean(dataset_means, axis=0)
    plt.plot(x_pos, mean_vals, color="black", linewidth=3, label="Overall Mean")
    plt.scatter(x_pos, mean_vals, color="black", s=100, zorder=3)

    xtick_labels = ["Shuffled Σ", "Meanfield Σ", "Real Σ"]
    plt.xticks(x_pos, xtick_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Mean R² Score", fontsize=20)
    plt.title(f"{label}: R² Across Σ Models", fontsize=22)
    plt.ylim(0.0, 0.8)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_name = f"{label}_mean_r2_comparison.svg"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, format='svg')
    plt.close()
    print(f"Saved: {save_path}")

    return r2_real_all, r2_null_all, r2_shuff_all

def plot_r2_comparisons_from_saved_csvs(comparison="null", csv_dir="./", save_dir="output"):
    assert comparison in ["null", "rand"], "comparison must be 'null' or 'rand'"

    all_real = []
    all_comp = []
    dataset_labels = []

    for file in os.listdir(csv_dir):
        if file.endswith("_results.csv"):
            df = pd.read_csv(os.path.join(csv_dir, file))
            comp_col = f"R2_{comparison}" if comparison == "null" else "R2_rand"
            if "R2_real" not in df or comp_col not in df:
                continue

            r2_real = df["R2_real"].dropna().values
            r2_other = df[comp_col].dropna().values

            all_real.extend(r2_real)
            all_comp.extend(r2_other)
            dataset_labels.extend([file.replace("_results.csv", "")] * len(r2_real))

    # Perform KS test
    ks_stat, p_value = ks_2samp(all_real, all_comp)
    print(f"\nKS test (Real Σ vs {'Shuffled Σ' if comparison == 'rand' else 'Null Σ'}):")
    print(f"   KS statistic = {ks_stat:.3f}")
    print(f"   p-value      = {p_value:.3e}")

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_real, bins=40, alpha=0.7, label="R² (Real Σ)", density=True)
    plt.hist(all_comp, bins=40, alpha=0.7, label=f"R² ({'Shuffled Σ' if comparison == 'rand' else 'Null Σ'})", density=True)
    plt.xlabel("R² Score", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title(f"R² across datasets: Real Σ vs {'Shuffled Σ' if comparison == 'rand' else 'Null Σ'}", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.legend(fontsize=20)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylim(1e-1, 1e2)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    fname = f"combined_r2_hist_{'shuffled' if comparison == 'rand' else 'null'}.svg"
    plt.savefig(os.path.join(save_dir, fname), format='svg')
    plt.show()

def print_global_test(name, real, baseline):
    if len(real) != len(baseline):
        print(f"Mismatch in lengths for {name}")
        return
    stat, p = wilcoxon(real, baseline, alternative="two-sided")
    diffs = real - baseline
    print(f"[{name}]")
    print(f"  - Δ = {np.round(diffs, 5)}")
    print(f"  - W = {stat:.3f}, p = {p:.6e}, n = {len(real)}")
