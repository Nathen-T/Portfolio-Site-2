# This file implements the L2,1-NMF (Non-negative Matrix Factorization with L2,1 norm regularization)
# It includes the L21NMF class and an experiment runner function.

import numpy as np
import json
from tools import load_data, add_noise, add_block_occlusion, evaluate_clustering

class L21NMF:
    """
    L2,1-NMF class for performing non-negative matrix factorization with L2,1 norm regularization.
    """
    def __init__(self, n_components=10, max_iter=1000, tol=1e-6, random_state=None):
        """
        Initialize the L21NMF model.
        
        Parameters:
        - n_components: Number of latent components (basis vectors).
        - max_iter: Maximum number of iterations for optimization.
        - tol: Tolerance for convergence (based on Frobenius norm difference).
        - random_state: Seed for random initialization of matrices.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.W = None  # Basis matrix (n_features x n_components)
        self.H = None  # Coefficient matrix (n_components x n_samples)
        self.reconstruction_err_ = []  # List to store L2,1 reconstruction errors per iteration
        
    def initialize_matrices(self, n_features, n_samples):
        """
        Initialize W and H matrices with random positive values.
        
        Parameters:
        - n_features: Number of features (rows in input matrix V).
        - n_samples: Number of samples (columns in input matrix V).
        """
        if self.random_state:
            np.random.seed(self.random_state)
        self.W = np.abs(np.random.randn(n_features, self.n_components)) + 1e-10  # Ensure positivity
        self.H = np.abs(np.random.randn(self.n_components, n_samples)) + 1e-10
        
    def compute_l21_norm(self, E):
        """
        Compute the L2,1 norm of the error matrix E.
        L2,1 norm is the sum of L2 norms over columns (samples).
        
        Parameters:
        - E: Error matrix (n_features x n_samples).
        
        Returns:
        - L2,1 norm value.
        """
        return np.sum(np.sqrt(np.sum(E**2, axis=0)))
    
    def update_diagonal_matrix(self, V, W, H):
        """
        Update the diagonal matrix D for L2,1 regularization.
        D is a diagonal matrix where each entry is 1 / ||error_per_sample||_2 for each sample.
        
        Parameters:
        - V: Original input matrix.
        - W: Current basis matrix.
        - H: Current coefficient matrix.
        
        Returns:
        - D: Diagonal matrix for weighting.
        """
        reconstruction = W @ H
        error_per_sample = V - reconstruction
        l2_norms = np.sqrt(np.sum(error_per_sample**2, axis=0))  # L2 norm per sample
        l2_norms = np.maximum(l2_norms, 1e-10)  # Avoid division by zero
        D = np.diag(1.0 / l2_norms)
        
        return D
    
    def fit_transform(self, V):
        """
        Fit the L21NMF model to the input matrix V and return the basis matrix W.
        Uses an alternating optimization approach with L2,1 regularization.
        
        Parameters:
        - V: Input matrix (n_features x n_samples).
        
        Returns:
        - W: Fitted basis matrix.
        """
        V = np.array(V, dtype=np.float64)
        n_features, n_samples = V.shape
        
        self.initialize_matrices(n_features, n_samples)
        
        W_prev = self.W.copy()
        H_prev = self.H.copy()
        
        # print(f"Starting L2,1-NMF optimization with {self.max_iter} max iterations...")
        
        for iteration in range(self.max_iter):
            # Update diagonal matrix D for L2,1 weighting
            D = self.update_diagonal_matrix(V, self.W, self.H)
            
            # Update W using the weighted least squares
            VDH_T = V @ D @ self.H.T
            WHDH_T = self.W @ self.H @ D @ self.H.T
            
            WHDH_T = np.maximum(WHDH_T, 1e-10)  # Avoid division by zero
            self.W *= VDH_T / WHDH_T
            
            self.W = np.maximum(self.W, 1e-10)  # Ensure non-negativity
            
            # Update H using the weighted least squares
            W_T_VD = self.W.T @ V @ D
            W_T_WHD = self.W.T @ self.W @ self.H @ D
            
            W_T_WHD = np.maximum(W_T_WHD, 1e-10)  # Avoid division by zero
            self.H *= W_T_VD / W_T_WHD
            
            self.H = np.maximum(self.H, 1e-10)  # Ensure non-negativity
            
            # Compute and store L2,1 reconstruction error
            reconstruction = self.W @ self.H
            l21_error = self.compute_l21_norm(V - reconstruction)
            self.reconstruction_err_.append(l21_error)
            
            # Check for convergence based on Frobenius norm differences
            W_diff = np.linalg.norm(self.W - W_prev, 'fro')
            H_diff = np.linalg.norm(self.H - H_prev, 'fro')
            
            # if iteration % 100 == 0:
            #     print(f"Iteration {iteration}: L2,1 error = {l21_error:.6f}, "
            #           f"W_diff = {W_diff:.6f}, H_diff = {H_diff:.6f}")
            
            if W_diff < self.tol and H_diff < self.tol:
                print(f"Converged at iteration {iteration}")
                break
                
            W_prev = self.W.copy()
            H_prev = self.H.copy()
            
        return self.W
        
    def reconstruct(self):
        """
        Reconstruct the input matrix using the fitted W and H.
        
        Returns:
        - Reconstructed matrix (n_features x n_samples).
        """
        return self.W @ self.H

def run_l21_experiment(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter):
    """
    Run L21NMF experiments on a dataset with specified noise, multiple runs, and evaluation metrics.
    Saves results to a JSON file.
    
    Parameters:
    - dataset: Name of the dataset ('ORL' or 'ExtendedYaleB').
    - noise: Type of noise ('salt_and_pepper' or 'block_occlusion').
    - n_runs: Number of experimental runs.
    - sample_ratio: Fraction of samples to use.
    - noise_params: Dictionary of noise parameters.
    - reduction_factor: Factor to reduce image dimensions.
    - max_iter: Maximum iterations for NMF.
    """
    # Set dataset-specific parameters
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
    n_components = np.unique(Y).shape[0]  # Number of classes as components
    
    # Initialize lists for metrics
    rre_scores = []  # Relative Reconstruction Error
    acc_scores = []  # Accuracy
    nmi_scores = []  # Normalized Mutual Information
    
    rng = np.random.default_rng(42)  # Random number generator for sampling

    # Create filename for results
    noise_vals = "_".join([f"{val}" for val in noise_params.values()])
    file_name = f"results_l21_{dataset}_{noise}_{noise_vals}.json"

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs} file: {file_name}")
        
        # Sample a subset of data
        n_samples_to_use = int(n_samples * sample_ratio)
        sample_indices = rng.choice(n_samples, n_samples_to_use, replace=False)
        X_sampled = X[:, sample_indices]
        Y_sampled = Y[sample_indices]
        
        # Add noise based on type
        if noise == 'salt_and_pepper':
            p = noise_params['p']
            r = noise_params['r']
            X_noisy = add_noise(X_sampled, p=p, r=r)
        elif noise == 'block_occlusion':
            block_width = noise_params['block_width']
            block_height = noise_params['block_height']
            X_noisy = add_block_occlusion(X_sampled, block_width, block_height, img_height=img_height, img_width=img_width)

        # Fit L21NMF model
        model = L21NMF(n_components=n_components, max_iter=max_iter, random_state=run)
        _ = model.fit_transform(X_noisy)
        X_reconstructed = model.reconstruct()
        
        # Evaluate metrics
        rre = np.linalg.norm(X_sampled - X_reconstructed, 'fro') / np.linalg.norm(X_sampled, 'fro')
        acc, nmi = evaluate_clustering(model.H, Y_sampled)
        
        # Store scores
        rre_scores.append(rre)
        acc_scores.append(acc)
        nmi_scores.append(nmi)
        
    # Save results to JSON
    results = {
        'algorithm': 'L2,1-NMF',
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
    # Main block for running a single experiment configuration
    dataset = "ORL"
    noise = 'salt_and_pepper'
    n_runs = 5
    sample_ratio = 0.9
    reduction_factor = 2
    max_iter = 50
    
    # Set noise parameters based on noise type
    if noise == 'salt_and_pepper':
        noise_params = {'p': 0.1, 'r': 0.5}
    elif noise == 'block_occlusion':
        noise_params = {'block_width': 20, 'block_height': 20}
    
    # Run the experiment (note: function returns values but they are commented out in the function)
    rre_scores, acc_scores, nmi_scores = run_l21_experiment(
        dataset=dataset,
        noise=noise,
        n_runs=n_runs,
        sample_ratio=sample_ratio,
        noise_params=noise_params,
        reduction_factor=reduction_factor,
        max_iter=max_iter
    )
    
    # Print summary statistics
    print(f"\nResults over {n_runs} runs:")
    print(f"RRE: Mean = {np.mean(rre_scores):.4f}, Std = {np.std(rre_scores):.4f}")
    print(f"ACC: Mean = {np.mean(acc_scores):.4f}, Std = {np.std(acc_scores):.4f}")
    print(f"NMI: Mean = {np.mean(nmi_scores):.4f}, Std = {np.std(nmi_scores):.4f}")