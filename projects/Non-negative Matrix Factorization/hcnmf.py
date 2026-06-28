# This file implements Hypersurface Cost-based Non-negative Matrix Factorization (HCNMF),
# It includes the core HCNMF algorithm, utility functions, and an experiment runner.

import os
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tools import add_block_occlusion, evaluate_clustering, load_data, visualize_noise_addition, add_noise


# ## Hypersurface Cost Based NMF

# In[44]:


def h_hypersurface(x):
    """
    Hypersurface cost function h(x) = sqrt(1 + x^2) - 1.
    This function is robust to outliers as it grows sublinearly for large x.
    
    Parameters:
    - x: Input value or array.
    
    Returns:
    - Computed h(x).
    """
    return np.sqrt(1.0 + x * x) - 1.0

def h_prime(x):
    """
    Derivative of the hypersurface cost function h'(x) = x / sqrt(1 + x^2).
    
    Parameters:
    - x: Input value or array.
    
    Returns:
    - Computed h'(x).
    """
    return x / np.sqrt(1.0 + x * x)

def hcnmf(
    V,
    rank,
    max_iter=3000,
    tol=1e-5,
    init_W=None,
    init_H=None,
    random_state=0,
    armijo_c=1e-4,
    armijo_beta=0.5,
    init_step_W=1.0,
    init_step_H=1.0,
    min_step=1e-16,
    nonneg_eps=0.0,
    verbose=False,
):
    """
    Perform Hypersurface Cost-based NMF (HCNMF) on matrix V.
    Uses gradient descent with Armijo line search for optimization.
    
    Parameters:
    - V: Input matrix (n_features x n_samples).
    - rank: Number of components (rank of factorization).
    - max_iter: Maximum iterations.
    - tol: Tolerance for convergence.
    - init_W, init_H: Initial matrices (optional).
    - random_state: Seed for initialization.
    - armijo_c, armijo_beta: Armijo rule parameters.
    - init_step_W, init_step_H: Initial step sizes.
    - min_step: Minimum step size.
    - nonneg_eps: Epsilon for non-negativity.
    - verbose: If True, print progress.
    
    Returns:
    - W: Basis matrix.
    - H: Coefficient matrix.
    - history: Dictionary with objective values.
    """
    V = np.asarray(V, dtype=np.float64)
    n, m = V.shape
    r = rank
    rng = np.random.default_rng(random_state)
    W = np.clip(rng.random((n, r)), 0, None) if init_W is None else np.clip(np.array(init_W, float, copy=True), 0, None)
    H = np.clip(rng.random((r, m)), 0, None) if init_H is None else np.clip(np.array(init_H, float, copy=True), 0, None)

    def objective(W, H):
        E = V - W @ H
        return np.sum(h_hypersurface(E))

    L_prev = objective(W, H)
    history = {"obj": [L_prev]}

    for it in range(1, max_iter + 1):
        # Compute residual E and derivative S = h'(E)
        WH = W @ H
        E = V - WH
        S = h_prime(E)

        # Gradients based on the hypersurface cost
        grad_W = -(S @ H.T)
        grad_H = -(W.T @ S)

        # Line search for W update
        step_W = init_step_W
        dW = -grad_W
        inner_W = np.sum(grad_W * dW)
        while step_W >= min_step:
            W_new = W + step_W * dW
            W_new = np.maximum(W_new, nonneg_eps) if nonneg_eps > 0 else np.clip(W_new, 0, None)
            L_try = objective(W_new, H)
            if L_try <= L_prev + armijo_c * step_W * inner_W:
                break
            step_W *= armijo_beta
        if step_W < min_step:
            W_new = W
            L_try = L_prev
        W = W_new
        L_mid = L_try

        # Update H with fresh residual and gradient
        WH = W @ H
        E = V - WH
        S = h_prime(E)
        grad_H = -(W.T @ S)
        dH = -grad_H
        inner_H = np.sum(grad_H * dH)

        step_H = init_step_H
        while step_H >= min_step:
            H_new = H + step_H * dH
            H_new = np.maximum(H_new, nonneg_eps) if nonneg_eps > 0 else np.clip(H_new, 0, None)
            L_try2 = objective(W, H_new)
            if L_try2 <= L_mid + armijo_c * step_H * inner_H:
                break
            step_H *= armijo_beta
        if step_H < min_step:
            H_new = H
            L_try2 = L_mid

        H = H_new
        L_curr = L_try2
        history["obj"].append(L_curr)

        rel_drop = (L_prev - L_curr) / max(1.0, L_prev)
        if verbose and (it % 50 == 0 or it == 1):
            print(f"iter {it:5d}  obj {L_curr:.6e}  rel_drop {rel_drop:.3e}")
        if rel_drop >= 0 and rel_drop < tol:
            if verbose:
                print(f"Converged at iter {it} with relative drop {rel_drop:.3e}")
            break
        L_prev = L_curr

    return W, H, history


def scale_01(X):
    """
    Scale input matrix X to [0,1] range, assuming 8-bit images if max > 1.
    
    Parameters:
    - X: Input matrix.
    
    Returns:
    - Scaled matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.max() > 1.0:
        X = X / 255.0
    X[X < 0] = 0.0
    return X

def run_hcnmf(V, rank, max_iter=2000, tol=1e-5, random_state=0,
              nonneg_eps=0.0, verbose=True,
              armijo_c=1e-4, armijo_beta=0.5, init_step_W=1.0, init_step_H=1.0):
    """
    Wrapper function to run HCNMF with default settings for robust face data.
    
    Parameters:
    - V: Input matrix.
    - rank: Number of components.
    - Other parameters: As in hcnmf.
    
    Returns:
    - W, H, history.
    """
    V = scale_01(V)
    W, H, history = hcnmf(
        V=V,
        rank=rank,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        armijo_c=armijo_c,
        armijo_beta=armijo_beta,
        init_step_W=init_step_W,
        init_step_H=init_step_H,
        nonneg_eps=nonneg_eps,
        verbose=verbose,
    )
    return W, H, history

def reconstruct(W, H):
    """
    Reconstruct the matrix from W and H.
    
    Parameters:
    - W: Basis matrix.
    - H: Coefficient matrix.
    
    Returns:
    - Reconstructed matrix V_hat = W @ H.
    """
    return W @ H

def rel_recon_error(V, V_hat, ord='fro'):
    """
    Compute relative reconstruction error.
    
    Parameters:
    - V: Original matrix.
    - V_hat: Reconstructed matrix.
    - ord: Norm order.
    
    Returns:
    - Relative error.
    """
    num = np.linalg.norm(V - V_hat, ord=ord)
    den = max(1.0, np.linalg.norm(V, ord=ord))
    return num / den

def explain_variance_fraction(V, V_hat):
    """
    Compute explained variance fraction (EVF).
    
    Parameters:
    - V: Original matrix.
    - V_hat: Reconstructed matrix.
    
    Returns:
    - EVF value.
    """
    num = np.linalg.norm(V - V_hat, 'fro')**2
    den = np.linalg.norm(V, 'fro')**2 + 1e-16
    return 1.0 - (num / den)



def run_hcnmf_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter):
    """
    Run HCNMF experiments on a dataset with noise, multiple runs, and save results.
    
    Parameters:
    - dataset: 'ORL' or 'ExtendedYaleB'.
    - noise: 'salt_and_pepper' or 'block_occlusion'.
    - n_runs: Number of runs.
    - sample_ratio: Fraction of samples.
    - noise_params: Noise parameters dict.
    - reduction_factor: Image reduction factor.
    - max_iter: Max iterations.
    """
    # Set dataset parameters
    if dataset == "ORL":
        root = 'data/ORL'
        img_width = 92 // reduction_factor
        img_height = 112 // reduction_factor
    elif dataset == "ExtendedYaleB":
        root = 'data/CroppedYaleB'
        img_width = 192 // reduction_factor
        img_height = 168 // reduction_factor
    
    # Load data
    X, Y = load_data(root=root, reduce=reduction_factor)
    _, n_samples = X.shape
    
    n_components = np.unique(Y).shape[0]
    
    # Initialize metric lists
    rre_scores = []
    acc_scores = []
    nmi_scores = []
    
    rng = np.random.default_rng(42)
    
    # Create filename
    noise_vals = "_".join([f"{val}" for val in noise_params.values()])
    file_name = f"results_HCNMF_{dataset}_{noise}_{noise_vals}.json"

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs} file: {file_name}")
        
        # Sample data
        n_samples_to_use = int(n_samples * sample_ratio)
        sample_indices = rng.choice(n_samples, n_samples_to_use, replace=False)
        X_sampled = X[:, sample_indices]
        Y_sampled = Y[sample_indices]
        
        # Add noise
        if noise == 'salt_and_pepper':
            p = noise_params['p']
            r = noise_params['r']
            X_noisy = add_noise(X_sampled, p=p, r=r)
        elif noise == 'block_occlusion':
            block_width = noise_params['block_width']
            block_height = noise_params['block_height']
            X_noisy = add_block_occlusion(X_sampled, block_width, block_height, img_height=img_height, img_width=img_width)

        # Run HCNMF
        W, H, history = run_hcnmf(
            V=X_noisy,
            rank=n_components,
            max_iter=max_iter,
            tol=1e-5,
            random_state=run,
            nonneg_eps=0.0,
            verbose=False
        )
        X_reconstructed = reconstruct(W, H)
        
        # Compute metrics
        rre = np.linalg.norm(X_sampled - X_reconstructed, 'fro') / np.linalg.norm(X_sampled, 'fro')
        acc, nmi = evaluate_clustering(H, Y_sampled)
        
        rre_scores.append(rre)
        acc_scores.append(acc)
        nmi_scores.append(nmi)
    
    # Save results
    results = {
        'algorithm': 'HCNMF',
        'dataset': dataset,
        'noise': noise,
        'noise_params': noise_params,
        'RRE': rre_scores,
        'ACC': acc_scores,
        'NMI': nmi_scores
    }
    with open(f"results/{file_name}", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to results/{file_name}")

    return rre_scores, acc_scores, nmi_scores


if __name__ == "__main__":
    # Main script to run experiments on multiple datasets and noise types
    dataset = "ORL"
    noise = 'salt_and_pepper'
    n_runs = 5
    sample_ratio = 0.9
    reduction_factor = 2

    # Load data for visualization (commented out in original)
    if dataset == "ORL":
        root = 'data/ORL'
        img_width = 92 // reduction_factor
        img_height = 112 // reduction_factor
    elif dataset == "ExtendedYaleB":
        root = 'data/CroppedYaleB'
        img_width = 192 // reduction_factor
        img_height = 168 // reduction_factor
    X, Y = load_data(root=root, reduce=reduction_factor)
    
    n_features, n_samples = X.shape
    n_components = np.unique(Y).shape[0]
    
    # visualize_noise_addition(X, img_width=img_width, img_height=img_height)
    
    results = []
    
    # Loop over datasets and noise configurations
    for dataset in ["ORL", "ExtendedYaleB"]:
        for noise in ['salt_and_pepper', 'block_occlusion']:
            if noise == 'salt_and_pepper':
                for p in [0.2, 0.4]:
                    for r in [0.5, 0.7]:
                        noise_params = {'p': p, 'r': r}

                        print(f"\nDataset: {dataset}, Noise: {noise}, Params: {noise_params}")
                        rre_scores, acc_scores, nmi_scores = run_hcnmf_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter=500)

                        results.append({
                            'dataset': dataset,
                            'noise': noise,
                            'noise_params': noise_params,
                            'RRE': rre_scores,
                            'ACC': acc_scores,
                            'NMI': nmi_scores
                        })
                        print(f"Results for {dataset}, {noise}, {noise_params}: {results[-1]}")

            elif noise == 'block_occlusion':
                for block_width in [10, 12]:
                    block_height = block_width

                    noise_params = {'block_width': block_width, 'block_height': block_height}

                    print(f"\nDataset: {dataset}, Noise: {noise}, Params: {noise_params}")
                    rre_scores, acc_scores, nmi_scores = run_hcnmf_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter=500)

                    results.append({
                        'dataset': dataset,
                        'noise': noise,
                        'noise_params': noise_params,
                        'RRE': rre_scores,
                        'ACC': acc_scores,
                        'NMI': nmi_scores
                    })
                    print(f"Results for {dataset}, {noise}, {noise_params}: {results[-1]}")

    # Save all results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to results.json")