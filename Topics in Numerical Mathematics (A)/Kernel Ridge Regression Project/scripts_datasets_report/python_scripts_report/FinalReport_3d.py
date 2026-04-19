# Setup
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

from functions import * 
from kernels import *

rng = np.random.default_rng(42)   #fix the randomness seed for reproducability

dataset_names = [
    ("pumadyn-8fm.data", "Pumadyn-8fm", 8000),     #1
]
kernels = [
      (linear_kernel_evaluation, "Linear kernel"),
      (second_order_kernel_evaluation, "2nd order"),
      (third_order_kernel_evaluation, "3rd order"),
      (rbf_kernel_evaluation, "RBF kernel"),
      (laplacian_kernel_evaluation, "Laplacian kernel"),
  ]

lambd = 10**(-3)
n = 100
ms = 10**np.linspace(0,5,100)    #m=1 to m=1000 
test_rate = 0  #not allowed by splitting functions, so manually set at 1 there 

repetitions = 1

for d_idx, (dataset_name, dataset_label, size) in enumerate(dataset_names):
    average_costs_K = np.zeros((len(ms), 5))
    average_costs_solve = np.zeros((len(ms), 5))
    
    for rep in range(repetitions):
        print(f"Testing {dataset_label} data set, currently at repetition {rep+1}")
        for m_idx, m in enumerate(ms):
            m = round(m)
            if m_idx in [3, 8, 13, 18, 23]:
                print(f"    Progress: {4*(m_idx+1)}%")
            xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_name, n, 1, rng, m)
            for k_idx, (kernel, kernel_name) in enumerate(kernels):                               
                start_K = time.time()
                K = build_kernel_matrix(xs_training, kernel)                
                done_K = time.time()
                average_costs_K[m_idx, k_idx] = 1/repetitions * (done_K - start_K)
                  
                start_solve = time.time()
                predictor = construct_y_predictor(K, xs_training, ys_training, lambd, kernel)
                done_solve = time.time()
                average_costs_solve[m_idx, k_idx] = 1/repetitions * (done_solve - start_solve)

    plt.title(f"Computational cost - constructing K - for {dataset_label}")
    plt.loglog(ms, average_costs_K[:,0], marker='.', label=f"Linear kernel")
    plt.loglog(ms, average_costs_K[:,1], marker='.', label=f"2nd order kernel")
    plt.loglog(ms, average_costs_K[:,2], marker='.', label=f"3rd order kernel")
    plt.loglog(ms, average_costs_K[:,3], marker='.', label=f"RBF kernel")
    plt.loglog(ms, average_costs_K[:,4], marker='.', label=f"Laplacian kernel")
    shift_1 = 0.3*average_costs_K[-1,2] / (ms[-1]) #compute shift such that O(n^2) fits nicely on the plot
    plt.loglog(ms, shift_1*ms,"--", color="black", label=r"$\mathcal{O}(m)$")
    plt.xlabel("Number of data features (m)")
    plt.ylabel("Run-time (s)")
    plt.ylim(bottom = 0.1*average_costs_K[0,4])
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()

    plt.title(f"Computational cost - solving system - for {dataset_label}")
    plt.loglog(ms, average_costs_solve[:,0], marker='.', label=f"Linear kernel")
    plt.loglog(ms, average_costs_solve[:,1], marker='.', label=f"2nd order kernel")
    plt.loglog(ms, average_costs_solve[:,2], marker='.', label=f"3rd order kernel")
    plt.loglog(ms, average_costs_solve[:,3], marker='.', label=f"RBF kernel")
    plt.loglog(ms, average_costs_solve[:,4], marker='.', label=f"Laplacian kernel")
    #shift_2 = 0.01*average_costs_K[-1,0] / (ms[-1]**3) #compute shift such that O(n^3) fits nicely on the plot
    #plt.loglog(ms, shift_2*(ms**3),"--", color="black", label=r"$\mathcal{O}(n^3)$")
    plt.xlabel("Number of data features (m)")
    plt.ylabel("Run-time (s)")
    plt.ylim(bottom = 0.1*average_costs_solve[0,4])
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()

    