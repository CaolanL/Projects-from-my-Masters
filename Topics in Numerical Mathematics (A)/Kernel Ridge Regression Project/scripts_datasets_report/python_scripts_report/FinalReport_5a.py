# Setup
import matplotlib.pyplot as plt
import numpy as np
from functions import * 
from kernels import *

rng = np.random.default_rng(42)   #fix the randomness seed for reproducability
test_rate = 0.3
n = 2000

dataset_names = [
    "pumadyn-8fm.data",     #0
    "pumadyn-8nh.data",     #1
    "abalone_clean.data",   #2
]



datasets = [    
    #(data file,       name,          kernel,                        lambda,   n,  test rate,  is the output an integer?)
    (dataset_names[0], "Pumadyn-8fm", second_order_kernel_evaluation, 10**(-5), n, test_rate, False),
    (dataset_names[1], "Pumadyn-8nh", third_order_kernel_evaluation, 10**(-5), n, test_rate, False),
    (dataset_names[2], "Abalone", rbf_kernel_evaluation, 10**(-6), n, test_rate, True)
]

sampling_methods = [
    #       scoring approach,          name    , p_factor
    (efficient_probabilities_uniform, "uniform", 0)
]

# Set-up
nr_epsilons = 100
epsilons = np.linspace(0.001, 0.1, nr_epsilons)  #fraction of total n columns to be sampled: s = epsilon * n

for ds_idx, (dataset_name, dataset_title, kernel, lambd, n, test_rate, integer) in enumerate(datasets):
    all_errors = np.zeros((len(sampling_methods), nr_epsilons)) #2D-array to store all errors
   
    xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_name, n, round(test_rate*n))

    K_exact = build_kernel_matrix(xs_training, kernel)
 
    # Construct appoximate kernel matrices
    for m_idx, (sampling_method, sampling_name, p_factor) in enumerate(sampling_methods):
        print(f"Started on {sampling_name}...")
        for e_idx, epsilon in enumerate(epsilons):
            s = max(round(n * epsilon), 1)
            p = max(round(p_factor * s), 1)          
            probabilities = sampling_method(xs_training, kernel, p, lambd, rng)
            columns_indices = rng.choice(n, s, replace=False, p=probabilities)       
            C, W = construct_C_and_W(xs_training, kernel, columns_indices, n, s)  
            U, Lambda = efficient_find_Khat(C, W, columns_indices)
            K_approx = U @ np.diag(Lambda) @ U.T
            all_errors[m_idx, e_idx] = (np.linalg.norm(K_exact - K_approx, ord = 'fro'))
        # Plotting
        sampled_columns = n * epsilons

    plt.title(f"Approximating $K$ using Uniform Sampling for {dataset_title}")
    plt.semilogy(sampled_columns, all_errors[0,:], label="Uniform")
    plt.xlabel(r"Number of Columns Sampled from $K$ to construct $\hat{K}$")
    plt.ylabel(r"Frobenius Error of the Approximation: ${||K - \hat{K}||_F}$")
    #plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()