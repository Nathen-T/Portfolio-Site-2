import os
import numpy as np
import matplotlib.pyplot as plt
from tools import load_data, add_noise, add_block_occlusion
from L21_nmf import L21NMF
from L1_Robust import L1RegularizedRobustNMF
from hcnmf import run_hcnmf, reconstruct
import time
from multiprocessing import Process

def visualize_results(dataset, noise, noise_params, reduction_factor, max_iter):
    """
    Visualize original, noisy, and reconstructed images for all three algorithms (L21, L1, HCNMF) in one figure.
    Saves the plot to the 'visualizations' folder.
    
    Parameters:
    - dataset: 'ORL' or 'ExtendedYaleB'.
    - noise: 'salt_and_pepper' or 'block_occlusion'.
    - noise_params: Dictionary of noise parameters.
    - reduction_factor: Image reduction factor.
    - max_iter: Maximum iterations for the algorithms.
    """
    
    # Log
    print(f"Visualizing {dataset} under {noise} noise and params {noise_params}")
    
    # Set dataset parameters
    if dataset == "ORL":
        root = 'data/ORL'
        img_width = 92 // reduction_factor
        img_height = 112 // reduction_factor
    elif dataset == "ExtendedYaleB":
        root = 'data/CroppedYaleB'
        img_height = 192 // reduction_factor
        img_width = 168 // reduction_factor
    
    # Load data
    X, Y = load_data(root=root, reduce=reduction_factor)
    _, n_samples = X.shape
    n_components = len(np.unique(Y))  # Number of unique classes
    
    # Sample a subset (use first sample for visualization)
    sample_ratio = 0.9
    n_samples_to_use = int(n_samples * sample_ratio)
    rng = np.random.default_rng(42)
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
    
    # Prepare reconstructions
    reconstructions = {}
    
    # L21
    model_l21 = L21NMF(n_components=n_components, max_iter=max_iter, random_state=42)
    model_l21.fit_transform(X_noisy)
    reconstructions['L21'] = model_l21.reconstruct()
    
    # L1
    model_l1 = L1RegularizedRobustNMF(n_components=n_components, max_iter=max_iter, random_state=42)
    model_l1.fit_transform(X_noisy)
    reconstructions['L1'] = model_l1.inverse_transform()
    
    # HCNMF
    W, H, _ = run_hcnmf(V=X_noisy, rank=n_components, max_iter=max_iter, random_state=42, verbose=False)
    reconstructions['HCNMF'] = reconstruct(W, H)
    
    # Create visualizations folder
    os.makedirs("visualizations", exist_ok=True)
    
    # Plot and save (using the first image as example)
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    # Original
    ax[0].imshow(X_sampled[:, 0].reshape(img_height, img_width), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Noisy
    ax[1].imshow(X_noisy[:, 0].reshape(img_height, img_width), cmap='gray')
    ax[1].set_title(f'Noisy Image ({noise})')
    ax[1].axis('off')
    
    # Reconstructed for each algorithm
    for i, alg in enumerate(['L21', 'L1', 'HCNMF'], start=2):
        ax[i].imshow(reconstructions[alg][:, 0].reshape(img_height, img_width), cmap='gray')
        ax[i].set_title(f'Reconstructed ({alg})')
        ax[i].axis('off')
    
    # Save the figure
    noise_str = "_".join([f"{k}_{v}" for k, v in noise_params.items()])
    filename = f"visualizations/{dataset}_{noise}_{noise_str}_all_algorithms.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    start = time.time()
    os.makedirs("visualizations", exist_ok=True)
    max_iter = 200
    reduction_factor = 2
    processes = []
    
    for dataset in ['ORL', 'ExtendedYaleB']:
        for noise in ['salt_and_pepper', 'block_occlusion']:
            if noise == 'salt_and_pepper':
                for p in [0.2, 0.4]:
                    for r in [0.5, 0.7]:
                        noise_params = {'p': p, 'r': r}
                        p_vis = Process(target=visualize_results, args=(dataset, noise, noise_params, reduction_factor, max_iter))
                        p_vis.start()
                        processes.append(p_vis)
            elif noise == 'block_occlusion':
                for block_width in [10, 12]:
                    block_height = block_width
                    noise_params = {'block_width': block_width, 'block_height': block_height}
                    p_vis = Process(target=visualize_results, args=(dataset, noise, noise_params, reduction_factor, max_iter))
                    p_vis.start()
                    processes.append(p_vis)
    
    print("All visualizations are running...")
    for p in processes:
        p.join()
    
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
