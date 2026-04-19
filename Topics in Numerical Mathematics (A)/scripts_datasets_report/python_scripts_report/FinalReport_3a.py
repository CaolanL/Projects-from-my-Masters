# Setup
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

from functions import * 
from kernels import *

rng = np.random.default_rng(42)   #fix the randomness seed for reproducability

dataset_names = [
    ("abalone_clean.data", "Abalone"),   #0
    ("pumadyn-8fm.data", "Pumadyn-8fm"),     #1
    ("pumadyn-8nh.data", "Pumadyn-8nh")     #2
]
kernels = [
      (linear_kernel_evaluation, "Linear kernel"),
      (second_order_kernel_evaluation, "2nd order"),
      (third_order_kernel_evaluation, "3rd order"),
      (rbf_kernel_evaluation, "RBF kernel"),
      (laplacian_kernel_evaluation, "Laplacian kernel"),
  ]

lambd = 10**(-3)
ns = np.linspace(1, 2000, 25)
test_rate = 0.3
repetitions = 3

for d_idx, (dataset_name, dataset_label) in enumerate(dataset_names):
    errors = np.zeros((len(ns), 5))
    for n_idx, n in enumerate(ns):
        n = int(round(n))
        print(f"n = {n}")

        for rep in range(repetitions):
            xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_name, n, max(round(test_rate*n),1) )
            for k_idx, (kernel, kernel_name) in enumerate(kernels):
                K = build_kernel_matrix(xs_training, kernel)
                predictor = construct_y_predictor(K, xs_training, ys_training, lambd, kernel)
                errors[n_idx, k_idx] =  1/repetitions * mean_squared_error(xs_testing, ys_testing, predictor)

    plt.title(f"Regression Error for {dataset_label}")
    plt.semilogy(ns, errors[:,0], marker='.', label=f"Linear kernel")
    plt.semilogy(ns, errors[:,1], marker='.', label=f"2nd order kernel")
    plt.semilogy(ns, errors[:,2], marker='.', label=f"3rd order kernel")
    plt.semilogy(ns, errors[:,3], marker='.', label=f"RBF kernel")
    plt.semilogy(ns, errors[:,4], marker='.', label=f"Laplacian kernel")
    plt.xlabel("Number of training data points (n)")
    plt.ylabel("Mean Squared Error")
    #if d_idx == 0: 
    #    plt.ylim((10**(0), 10**(1)))
    #if d_idx == 1:
    #    plt.ylim((10**(0), 2*10**(1)))
    #if d_idx == 2: 
    #    plt.ylim((10**(1), 10**(2)))

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()

    