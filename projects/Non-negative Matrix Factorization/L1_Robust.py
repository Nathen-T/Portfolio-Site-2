# This file implements L1-Regularized Robust Non-negative Matrix Factorization (NMF) for image denoising.
# It minimizes the L1 reconstruction error with L1 regularization on W and H, using Huber loss for robustness.
# Includes the L1RegularizedRobustNMF class and an experiment runner function.

import numpy as np
from tools import load_data, add_noise, add_block_occlusion, evaluate_clustering
import json

class L1RegularizedRobustNMF:
    """
    Simplified L1 Robust NMF with L1 Regularization
    
    Solves: min ||V - WH||_1 + λ1||W||_1 + λ2||H||_1
    subject to: W ≥ 0, H ≥ 0
    """
    
    def __init__(self, n_components=50, lambda1=0.0001, lambda2=0.0001, 
                 max_iter=1000, tol=1e-6, step_size=0.0001, random_state=None):
        
        """
        Initialize the L1RegularizedRobustNMF model.

        Parameters:
        - n_components: Number of latent components (basis vectors).
        - max_iter: Maximum number of iterations for optimization.
        - lambda1: Regularization parameter for W.
        - lambda2: Regularization parameter for H.
        - step_size: Fixed step size for gradient descent updates.
        - tol: Tolerance for convergence (based on Frobenius norm difference).
        - random_state: Seed for random initialization of matrices.
        """

        self.n_components = n_components
        self.lambda1 = lambda1 
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.random_state = random_state
        self.eps = 1e-10
        
        self.W = None
        self.H = None
    
    def _initialize(self, V):
        """
        Initialize W and H matrices with SVD.

        SVD is used for a more informed initialization, followed by enforcing non-negativity.
        
        Parameters:
        - n_features: Number of features (rows in input matrix V).
        - n_samples: Number of samples (columns in input matrix V).
        - rank: Effective rank for initialization (min(n_components, rank of V)).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_features, n_samples = V.shape
        
        # SVD-based initialization
        U, s, Vt = np.linalg.svd(V + self.eps, full_matrices=False)
        rank = min(self.n_components, len(s))
        
        # Initialize using truncated SVD with positivity constraint
        self.W = np.maximum(U[:, :rank] @ np.diag(np.sqrt(s[:rank])), self.eps)
        self.H = np.maximum(np.diag(np.sqrt(s[:rank])) @ Vt[:rank, :], self.eps)
        
        # Pad with small random values if needed
        if self.n_components > rank:
            data_scale = np.mean(V) / self.n_components
            W_extra = np.random.exponential(data_scale, (n_features, self.n_components - rank))
            H_extra = np.random.exponential(data_scale, (self.n_components - rank, n_samples))
            self.W = np.hstack([self.W, W_extra])
            self.H = np.vstack([self.H, H_extra])
    
    def _l1_loss_gradient(self, residual):
        """L1 loss gradient (sign function)

        Applies the sign function to the residual.

        Sign Function:
        - For x > 0, sign(x) = 1
        - For x < 0, sign(x) = -1
        - For x = 0, sign(x) = 0

        Parameters:
        - residual: V - WH
        Returns:
        Returns:
        - Gradient of the L1 loss with respect to the input residual.
        """
        return np.sign(residual)
    
    def _compute_cost(self, V):
        """Compute L1 objective function"""
        WH = self.W @ self.H
        reconstruction = np.sum(np.abs(V - WH))
        reg_W = self.lambda1 * np.sum(self.W)
        reg_H = self.lambda2 * np.sum(self.H)
        return reconstruction + reg_W + reg_H
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for L1 regularization

        Applies the sign function and shrinks values towards zero by the threshold.
        
        Parameters:
        - x: Input matrix.
        - threshold: Threshold value.
        Returns:
        - Soft-thresholded matrix.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _update_W(self, V):
        """Update W using fixed step size gradient descent

        Gradient step followed by soft thresholding for L1 regularization.
        
        Parameters:
        - V: Input data matrix.
        Returns:
        - Updated W matrix.
        """
        WH = self.W @ self.H
        residual = V - WH
        
        # L1 reconstruction gradient
        grad_recon = -self._l1_loss_gradient(residual) @ self.H.T
        
        # Simple gradient step
        W_temp = self.W - self.step_size * grad_recon
        
        # Apply soft thresholding for L1 regularization
        threshold = self.step_size * self.lambda1
        self.W = self._soft_threshold(W_temp, threshold)
        
        # Ensure positivity
        self.W = np.maximum(self.W, self.eps)
    
    def _update_H(self, V):
        """Update H using fixed step size gradient descent

        Gradient step followed by soft thresholding for L1 regularization.
        
        Parameters:
        - V: Input data matrix.
        Returns:
        - Updated H matrix.
        """
        WH = self.W @ self.H
        residual = V - WH
        
        # L1 reconstruction gradient
        grad_recon = -self.W.T @ self._l1_loss_gradient(residual)
        
        # Simple gradient step
        H_temp = self.H - self.step_size * grad_recon
        
        # Apply soft thresholding for L1 regularization
        threshold = self.step_size * self.lambda2
        self.H = self._soft_threshold(H_temp, threshold)
        
        # Ensure positivity
        self.H = np.maximum(self.H, self.eps)
    
    def fit(self, V, verbose=False):
        """Fit the L1 robust NMF model

        Fits the model to the input data matrix V using alternating updates for W and H.

        Parameters:
        - V: Input data matrix (n x m).
        - verbose: If True, print progress messages.
        Returns:
        - self: Fitted model.
        """
        V = np.asarray(V, dtype=np.float64)
        if V.min() < 0:
            raise ValueError("Input matrix must be non-negative")
        
        self._initialize(V)
        
        prev_cost = self._compute_cost(V)
        if verbose:
            print(f"Initial cost: {prev_cost:.4f}")
            print(f"Data shape: {V.shape}, Components: {self.n_components}")
            print(f"Step size: {self.step_size}, Regularization: λ1={self.lambda1}, λ2={self.lambda2}")
        
        for iteration in range(self.max_iter):
            # Store previous matrices for convergence check
            W_prev = self.W.copy()
            H_prev = self.H.copy()
            
            # Alternating updates
            self._update_W(V)
            self._update_H(V)
            
            # Check convergence
            current_cost = self._compute_cost(V)
            
            if verbose and iteration % 100 == 0:
                recon_error = np.sum(np.abs(V - self.W @ self.H))
                print(f"Iter {iteration}: Cost = {current_cost:.4f}, Recon = {recon_error:.4f}")
            
            # Cost-based convergence
            if prev_cost > 0 and abs(prev_cost - current_cost) / prev_cost < self.tol:
                if verbose:
                    print(f"Converged (cost) at iteration {iteration}")
                break
            
            # Parameter-based convergence
            W_change = np.linalg.norm(self.W - W_prev) / (np.linalg.norm(W_prev) + self.eps)
            H_change = np.linalg.norm(self.H - H_prev) / (np.linalg.norm(H_prev) + self.eps)
            
            if W_change < self.tol and H_change < self.tol:
                if verbose:
                    print(f"Converged (parameters) at iteration {iteration}")
                break
            
            prev_cost = current_cost
        
        return self
    
    def transform(self, V, max_iter=300):
        """Transform new data to latent space
        
        Parameters:
        - V: New data matrix to transform.
        - max_iter: Maximum iterations for optimizing H.
        Returns:
        - H: Latent representation of V.
        
        """
        if self.W is None:
            raise ValueError("Model must be fitted first")
        
        V = np.asarray(V, dtype=np.float64)
        n_features, n_samples = V.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state + 1)
        
        # Initialize H
        data_mean = np.mean(V)
        data_std = np.std(V)
        H = np.random.normal(data_mean/self.n_components, 
                           data_std/(6*self.n_components), 
                           (self.n_components, n_samples))
        H = np.maximum(H, self.eps)
        
        # Optimize H with fixed W
        for iteration in range(max_iter):
            H_prev = H.copy()
            
            WH = self.W @ H
            residual = V - WH
            
            # L1 reconstruction gradient
            grad_recon = -self.W.T @ self._l1_loss_gradient(residual)
            
            # Simple gradient step with soft thresholding
            H_temp = H - self.step_size * grad_recon
            threshold = self.step_size * self.lambda2
            H = self._soft_threshold(H_temp, threshold)
            H = np.maximum(H, self.eps)
            
            # Check convergence
            H_change = np.linalg.norm(H - H_prev) / (np.linalg.norm(H_prev) + self.eps)
            if H_change < self.tol:
                break
        
        return H
    
    def fit_transform(self, V, verbose=False):
        """Fit model and return latent representation"""
        self.fit(V, verbose=verbose)
        return self.H.copy()
    
    def inverse_transform(self, H=None):
        """Reconstruct data from latent representation"""
        if self.W is None:
            raise ValueError("Model must be fitted first")
        if H is None:
            H = self.H
        return self.W @ H
    
    def reconstruction_error(self, V, metric='l1'):
        """Compute reconstruction error"""
        if self.W is None or self.H is None:
            raise ValueError("Model must be fitted first")
        
        H = self.transform(V)
        WH = self.W @ H
        residual = V - WH

        if metric == 'l1':
            return np.sum(np.abs(residual))
        elif metric == 'l2':
            return np.sqrt(np.sum(residual**2))
        elif metric == 'frobenius':
            return np.linalg.norm(residual, 'fro')
        else:
            raise ValueError("metric must be 'l1', 'l2', or 'frobenius'")

def run_l1_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter):
    """
    Run L1-Regularized Robust NMF experiments on a dataset with noise, multiple runs, and save results.
    
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
    rre_scores = []  # Relative Reconstruction Error
    acc_scores = []  # Accuracy
    nmi_scores = []  # Normalized Mutual Information
    
    rng = np.random.default_rng(42)

    # Create filename
    noise_vals = "_".join([f"{val}" for val in noise_params.values()])
    file_name = f"results_l1_{dataset}_{noise}_{noise_vals}.json"

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

        # Fit model
        model = L1RegularizedRobustNMF(n_components=n_components, max_iter=max_iter, random_state=run)
        _ = model.fit_transform(X_noisy)
        X_reconstructed = model.inverse_transform()
        
        # Compute metrics
        rre = np.linalg.norm(X_sampled - X_reconstructed, 'fro') / np.linalg.norm(X_sampled, 'fro')  # Relative Reconstruction Error
        acc, nmi = evaluate_clustering(model.H, Y_sampled)
        
        rre_scores.append(rre)
        acc_scores.append(acc)
        nmi_scores.append(nmi)
    
    # Save results
    results = {
        'algorithm': 'L1-Robust-NMF',
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
    # Main script for running a single L1 experiment
    dataset = "ORL"
    noise = 'salt_and_pepper'
    n_runs = 5
    sample_ratio = 0.9
    reduction_factor = 2
    noise_params = {'p': 0.1, 'r': 0.5}  # For salt and pepper noise
    # noise_params = {'block_width': 20, 'block_height': 20}  # For block occlusion
    max_iter = 10
    
    rre_scores, acc_scores, nmi_scores = run_l1_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter)
    
    print("\nFinal Results:")
    print(f"RRE: Mean={np.mean(rre_scores):.4f}, Std={np.std(rre_scores):.4f}")
    print(f"ACC: Mean={np.mean(acc_scores):.4f}, Std={np.std(acc_scores):.4f}")
    print(f"NMI: Mean={np.mean(nmi_scores):.4f}, Std={np.std(nmi_scores):.4f}")