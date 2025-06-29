# this is the main code to run the inference. you run it for 
# 1) real covariance sigma 2) shuffled 3) mean-field and 4) znib distribution

import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import pinv
import os
import pickle

import numpy as np
from scipy.stats import nbinom
from scipy.sparse import issparse

from src.preprocess import get_data


def sample_zinb(mean, var, zero_prob, size):
    """Sample from ZINB with given mean, variance, and zero inflation."""
    if var <= mean:
        var = mean + 1e-3  # prevent invalid NB parameters

    # Compute NB parameters
    p = mean / var
    r = mean**2 / (var - mean)

    # Sample from NB
    nb_samples = nbinom.rvs(n=r, p=p, size=size)

    # Apply zero inflation
    zeros = np.random.rand(size) < zero_prob
    nb_samples[zeros] = 0
    return nb_samples

def compute_zinb_covariance(X0, n_cells=None, seed=42):
    """Generate synthetic data from ZINB with random scaling of variance."""
    if issparse(X0):
        X0 = X0.toarray()

    np.random.seed(seed)
    n_obs, n_genes = X0.shape
    n_cells = n_obs if n_cells is None else n_cells

    mu = X0.mean(axis=0)
    var_emp = X0.var(axis=0)
    zero_prob = (X0 == 0).mean(axis=0)

    # Draw random scaling factor for each gene
    scaling_factors = np.random.uniform(-3, 3, size=n_genes)
    scaling_factors = 10**scaling_factors

    n_half = n_genes // 2
    # Sample half from [-3, -2], half from [2, 3]
    low_range = np.random.uniform(-1, -.5, size=n_half)
    high_range = np.random.uniform(.5, 1, size=n_genes - n_half)
    # Combine and shuffle
    combined = np.concatenate([low_range, high_range])
    np.random.shuffle(combined)
    scaling_factors = 10**combined
    
    target_var = scaling_factors * var_emp

    synthetic_data = np.zeros((n_cells, n_genes))

    for g in range(n_genes):
        synthetic_data[:, g] = sample_zinb(mu[g], target_var[g], zero_prob[g], n_cells)

    Sigma_zinb = np.cov(synthetic_data, rowvar=False)
    return Sigma_zinb, synthetic_data, scaling_factors



def get_gene_index(gene_name, gene_names):
    indices = np.where(gene_names == gene_name)[0]
    return int(indices[0]) if indices.size > 0 else None

def compute_covariance(X0):
    return np.cov(X0, rowvar=False)

def compute_average_response(X0, X1):
    return X1.mean(axis=0) - X0.mean(axis=0)

def compute_sparse_perturbation(Sigma_inv, delta_X, gene_index, top_k=100):
    u_hat = Sigma_inv @ delta_X
    sorted_indices = np.argsort(np.abs(u_hat))[::-1]  # descending order
    top_indices = sorted_indices[:top_k].tolist()

    if gene_index in top_indices:
        rank = top_indices.index(gene_index)
        print(f"True perturbation is in top-{top_k}, index = {gene_index}, rank in top_k = {rank}")
    else:
        print(f"True perturbation NOT in top-{top_k}, forcibly included at index = {gene_index}")
        top_indices[0] = gene_index  # force inclusion

    u_sparse = np.zeros_like(u_hat)
    u_sparse[top_indices] = u_hat[top_indices]
    return u_sparse, sorted(top_indices)

def run_mcmc_horseshoe_learnable_sigma(Sigma_sub, delta_X_sub, draws=1000, tune=1000):
    G = Sigma_sub.shape[0]
    print(f"\n[MCMC] Running NUTS with learnable σ_obs on {G} genes...")

    with pm.Model() as model:
        # Horseshoe prior: non-centered
        lambda_ = pm.HalfCauchy("lambda", beta=1.0, shape=G)
        # log_tau = pm.Normal("log_tau", mu=-2, sigma=2)
        # tau = pm.Deterministic("tau", pm.math.exp(log_tau))
        log_tau = pm.Normal("log_tau", mu=-4, sigma=1)  # stronger global shrinkage
        tau = pm.Deterministic("tau", pm.math.exp(log_tau))

        
        z = pm.Normal("z", 0, 1, shape=G)
        u = pm.Deterministic("u", z * tau * lambda_)

        # Learnable observation noise
        # sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        log_sigma = pm.Normal("log_sigma_obs", mu=-2, sigma=2)
        sigma_obs = pm.Deterministic("sigma_obs", pm.math.exp(log_sigma))

        # Likelihood
        mu_x = pm.math.dot(Sigma_sub, u)
        obs = pm.Normal("obs", mu=mu_x, sigma=sigma_obs, observed=delta_X_sub)

        # MCMC inference
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.95,
            max_treedepth=10,
            chains=8,
            cores=8,
            progressbar=True
        )

    print("[MCMC] Done. Learned σ_obs samples available.\n")
    return trace


def run_bayesian_lasso(Sigma_sub, delta_X_sub, draws=1000, tune=1000):
    G = Sigma_sub.shape[0]
    print(f"\n[MCMC] Running Bayesian Lasso on {G} genes...")
    with pm.Model() as model:
        # Laplace prior promotes sparsity
        b = pm.HalfNormal("b", sigma=1.0)
        u = pm.Laplace("u", mu=0, b=b, shape=G)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        mu_x = pm.math.dot(Sigma_sub, u)
        obs = pm.Normal("obs", mu=mu_x, sigma=sigma_obs, observed=delta_X_sub)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.95,
            chains=2,
            cores=2,
            progressbar=True
        )
    print("[MCMC] Done (Bayesian Lasso).\n")
    return trace

def save_u_samples_summary(data_path, output_dir="u_samples_summaries", mode="u_samples_summaries"):
    os.makedirs(output_dir, exist_ok=True)

    adata, X0, X1 = get_data(0, data_path)
    gene_names = np.array(adata.var_names.tolist())

    if mode=="u_samples_summaries":
        Sigma = compute_covariance(X0)
    elif mode=="u_samples_summaries_shuff_X0":
        print("Shuffling each gene independently...")
        X0_shuffled = X0.copy()
        for g in range(X0.shape[1]):
            np.random.shuffle(X0_shuffled[:, g])  # shuffle expression of gene g across cells
        # ------------------------------
        # Step 3: Compute Σ_S and Σ_S⁻¹ from shuffled matrix
        # ------------------------------
        print("Computing shuffled covariance...")
        Sigma = compute_covariance(X0_shuffled)
        print("Done computing Σ_S and Σ_S⁻¹.")
    elif mode=="u_samples_summaries_shuff_Sigma":
        # Assume Sigma is your covariance matrix
        Sigma0 = compute_covariance(X0)
        
        # Flatten to vector, shuffle, and reshape
        flat = Sigma0.flatten()
        np.random.shuffle(flat)
        shuffled = flat.reshape(Sigma0.shape)
        
        # Make symmetric: average with its transpose
        Sigma = (shuffled + shuffled.T) / 2


    elif mode=="u_samples_summaries_shuff_Sigma_zinb":

        Sigma, zinb_data, scaling_factors = compute_zinb_covariance(X0)

   
    Sigma_inv = pinv(Sigma)

    perturbation_counts = adata.obs['perturbation'].value_counts()
    perturbations = [p for p in perturbation_counts.index if p != 'control']
    all_rows = []

    for pert in perturbations:
        gene_index = get_gene_index(pert, gene_names)
        if gene_index is None:
            continue

        X1 = adata[adata.obs['perturbation'] == pert].X
        X1 = X1.toarray() if hasattr(X1, "toarray") else X1
        delta_X = compute_average_response(X0, X1)
        u_sparse, sparse_indices = compute_sparse_perturbation(Sigma_inv, delta_X, gene_index, top_k=200)

        Sigma_sub = Sigma[np.ix_(sparse_indices, sparse_indices)]
        delta_X_sub = delta_X[sparse_indices]

        try:
            trace_hs = run_mcmc_horseshoe_learnable_sigma(Sigma_sub, delta_X_sub, draws=1000, tune=1000)
            # trace_hs = run_bayesian_lasso(Sigma_sub, delta_X_sub, draws=1000, tune=1000)
        except Exception as e:
            print(f"[SKIP] {pert} due to MCMC error: {e}")
            continue

        for model_name, trace in zip(['Horseshoe'], [trace_hs]):
            u_samples = trace.posterior['u'].stack(sample=("chain", "draw")).values
            u_mean_local = np.mean(u_samples, axis=1)
            u_std_local = np.std(u_samples, axis=1)
            # snr = np.abs(u_mean_local) / (u_std_local + 1e-6)

            # ci_lower = np.percentile(u_samples, 2.5, axis=1)
            # ci_upper = np.percentile(u_samples, 97.5, axis=1)
            # ci_excludes_0 = (ci_lower > 0) | (ci_upper < 0)

            # pip_local =  (ci_excludes_0.astype(float))
            pip_local = np.mean(np.abs(u_samples) > 0.05, axis=1)

            for j, idx in enumerate(sparse_indices):
                row = {
                    "Gene": gene_names[idx],
                    "Perturbation": pert,
                    "Model": model_name,
                    "IsTruePerturbation": int(idx == gene_index),
                    "PIP": pip_local[j],
                    "U_Mean": u_mean_local[j],
                    "U_Std": u_std_local[j],
                    "U_Samples": u_samples[j].tolist() if idx == gene_index else None
                }
                all_rows.append(row)

            # Plot PIP vs gene index
            pip_full = np.zeros(len(gene_names))
            for idx, p in zip(sparse_indices, pip_local):
                pip_full[idx] = p

            plt.figure(figsize=(12, 4))
            plt.scatter(range(len(gene_names)), pip_full, alpha=0.5)
            if pip_full[gene_index] > 0:
                plt.scatter(gene_index, pip_full[gene_index], color='red', s=50, label=f"True Perturbation: {pert}")
            plt.axhline(0.)
            plt.title(f"[{model_name}] PIP vs. Genes for Perturbation: {pert}")
            plt.xlabel("Gene Index")
            plt.ylabel("Posterior Inclusion Probability")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Plot u mean ± std for sparse subset
            plt.figure(figsize=(12, 4))
            plt.errorbar(
                sparse_indices, u_mean_local, yerr=u_std_local,
                fmt='o', alpha=0.7, capsize=3, label='Posterior mean ± std'
            )
            if gene_index in sparse_indices:
                i = sparse_indices.index(gene_index)
                plt.scatter([sparse_indices[i]], [u_mean_local[i]], color='red', s=50, label=f"True Perturbation: {pert}")
            plt.title(f"[{model_name}] Posterior Mean and Uncertainty of $u$ for Perturbation: {pert}")
            plt.xlabel("Gene Index")
            plt.ylabel("Posterior Mean of $u$")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # Histogram for true perturbation's posterior
            if gene_index in sparse_indices:
                i = sparse_indices.index(gene_index)
                plt.figure(figsize=(8, 4))
                plt.hist(u_samples[i], bins=50, density=True, alpha=0.7)
                plt.axvline(0, color='black', linestyle='--')
                plt.title(f"[{model_name}] Posterior of $u_{{{pert}}}$ (True Perturbed Gene)")
                plt.xlabel("Value of $u$")
                plt.ylabel("Density")
                plt.tight_layout()
                plt.yscale('log')
                plt.show()

    summary_df = pd.DataFrame(all_rows)
    o = os.path.join(output_dir, mode)
    os.makedirs(o, exist_ok=True)
    save_path = os.path.join(o, os.path.basename(data_path).replace(".h5ad", "_usamples_with_lr.pkl"))
    with open(save_path, "wb") as f:
        pickle.dump(summary_df, f)

    print(f"Saved u-sample summary to: {save_path}")
    return summary_df

def load_u_samples_summary(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
