# Final analysis, using only the 3 feasible scoring approaches. 
# -1: Regression accuracy
# -2: Run-time


# Setup
import matplotlib.pyplot as plt
import numpy as np
import time
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
    (efficient_probabilities_uniform, "uniform", 0),    #Dummy, to prevent first-iteration run-time issues like before
    (efficient_probabilities_uniform, "uniform", 0),
    (efficient_probabilities_diagonal, "diagonal", 0),
    (efficient_probabilities_approx_lambda_ridge, "approx lambda ridgee", 0.1)
]

# Set-up
nr_epsilons = 50
epsilons = np.linspace(0.002, 0.2, nr_epsilons)  #fraction of total n columns to be sampled: s = epsilon * n

for ds_idx, (dataset_name, dataset_title, kernel, lambd, n, test_rate, integer) in enumerate(datasets):
    all_errors_y = np.zeros((len(sampling_methods), nr_epsilons)) #2D-array to store all errors
    all_costs = np.zeros((len(sampling_methods), nr_epsilons)) #2D-array to store all errors
    
    xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_name, n, round(test_rate*n))

    # Construct appoximate kernel matrices
    for m_idx, (sampling_method, sampling_name, p_factor) in enumerate(sampling_methods):
        print(f"Started on {sampling_name}...")
        for e_idx, epsilon in enumerate(epsilons):
            s = max(round(n * epsilon), 1)
            p = max(round(p_factor * s), 1)          
            
            timing_start = time.time()
            probabilities = sampling_method(xs_training, kernel, p, lambd, rng)
            columns_indices = rng.choice(n, s, replace=False, p=probabilities)       
            C, W = construct_C_and_W(xs_training, kernel, columns_indices, n, s)  
            U, Lambda = efficient_find_Khat(C, W, columns_indices)
            predictor = efficient_construct_y_predictor(U, Lambda, xs_training, ys_training, lambd, kernel)
            timing_stop = time.time()
            #Compute the *Relative* Frobenius norm of the difference
            all_errors_y[m_idx, e_idx] =  mean_squared_error(xs_testing, ys_testing, predictor)
            all_costs[m_idx, e_idx] = timing_stop - timing_start

    print(f"Started on full KRR")
    timing_start_KRR = time.time()
    K_exact = build_kernel_matrix(xs_training, kernel)    
    predictor = construct_y_predictor(K_exact, xs_training, ys_training, lambd, kernel)
    timing_stop_KRR = time.time()
    all_costs[0,:] = timing_stop_KRR - timing_start_KRR
    all_errors_y[0, :] = mean_squared_error(xs_testing, ys_testing, predictor)

    # Plotting
    sampled_columns = n * epsilons
    plt.title(f"Regression Accuracy for {dataset_title}")
    plt.semilogy(sampled_columns, all_errors_y[0,:], '--', color = 'black', label="Classical")
    plt.semilogy(sampled_columns, all_errors_y[1,:], marker='.', label="Random, uniform")
    plt.semilogy(sampled_columns, all_errors_y[2,:], marker='.', label="Random, diagonal")
    plt.semilogy(sampled_columns, all_errors_y[3,:], marker='.', label="Random, approx lambda-ridge")
    plt.xlabel(r"Number of Columns Sampled from $K$ to construct $\hat{K}$")
    plt.ylabel(r"Mean Squared Error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()

    # Plotting
    plt.title(f"Computational Cost for {dataset_title}")
    plt.semilogy(sampled_columns, all_costs[0,:], '--', color = 'black', label="Classical")
    plt.semilogy(sampled_columns, all_costs[1,:], marker='.', label="Random, uniform")
    plt.semilogy(sampled_columns, all_costs[2,:], marker='.', label="Random, diagonal")
    plt.semilogy(sampled_columns, all_costs[3,:], marker='.', label="Random, approx lambda-ridge")
    plt.xlabel(r"Number of Columns Sampled from $K$ to construct $\hat{K}$")
    plt.ylabel(r"Run-time (s)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()