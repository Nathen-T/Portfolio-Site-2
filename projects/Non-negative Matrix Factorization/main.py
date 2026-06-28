# This script runs parallel experiments for three NMF algorithms (L21-NMF, L1-Robust-NMF, HCNMF)
# on ORL and ExtendedYaleB datasets with various noise configurations.
# It uses multiprocessing to execute experiments concurrently for efficiency.

from L21_nmf import run_l21_experiment  # Import L21-NMF experiment function
from L1_Robust import run_l1_experiment  # Import L1-Robust-NMF experiment function
from hcnmf import run_hcnmf_experiment  # Import HCNMF experiment function
import time  # For timing the execution
from multiprocessing import Process  # For parallel processing
import os  # For creating directories


if __name__ == "__main__":
    start = time.time()  # Record start time
    os.makedirs("results", exist_ok=True)  # Create results directory if it doesn't exist
    n_runs = 5  # Number of experimental runs per configuration
    sample_ratio = 0.9  # Fraction of samples to use
    reduction_factor = 2  # Factor to reduce image dimensions
    max_iter = 200  # Maximum iterations for NMF algorithms
    
    processes = []  # List to hold process objects
    
    # Loop over datasets and noise types
    for dataset in ["ORL", "ExtendedYaleB"]:
        for noise in ['salt_and_pepper', 'block_occlusion']:
            if noise == 'salt_and_pepper':
                # Iterate over salt-and-pepper noise parameters
                for p in [0.2, 0.4]:
                    for r in [0.5, 0.7]:
                        noise_params = {'p': p, 'r': r}

                        # Create processes for each algorithm
                        p1 = Process(target=run_l21_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                        p2 = Process(target=run_l1_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                        p3 = Process(target=run_hcnmf_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                        
                        # Start the processes
                        p1.start()
                        p2.start()
                        p3.start()
                        
                        # Add to processes list for joining later
                        processes.extend([p1, p2, p3])
            elif noise == 'block_occlusion':
                # Iterate over block occlusion parameters
                for block_width in [10, 12]:
                    block_height = block_width

                    noise_params = {'block_width': block_width, 'block_height': block_height}

                    # Create and start processes for each algorithm
                    p1 = Process(target=run_l21_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                    p2 = Process(target=run_l1_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                    p3 = Process(target=run_hcnmf_experiment, args=(dataset, noise, n_runs, sample_ratio, noise_params, reduction_factor, max_iter))
                    
                    p1.start()
                    p2.start()
                    p3.start()
                    
                    processes.extend([p1, p2, p3])

    print("All experiments are running...")  # Inform user that processes have started
    for p in processes:
        p.join()  # Wait for all processes to complete

    elapsed = time.time() - start  # Calculate elapsed time

    print(f"Elapsed time: {elapsed/60:.2f} minutes")  # Print total time in minutes