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


lambdas = 10**np.linspace(-12, 2, 25)
n = 2000
test_rate = 0.3
repetitions = 3

for d_idx, (dataset_name, dataset_label) in enumerate(dataset_names):
    errors = np.zeros((len(lambdas), 5))
    for l_idx, lambd in enumerate(lambdas):
        print(f"lambda = {lambd}")
        for rep in range(repetitions):
            xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_name, n, round(test_rate*n))
            for k_idx, (kernel, kernel_name) in enumerate(kernels):
                K = build_kernel_matrix(xs_training, kernel)
                predictor = construct_y_predictor(K, xs_training, ys_training, lambd, kernel)
                errors[l_idx, k_idx] = 1/repetitions * mean_squared_error(xs_testing, ys_testing, predictor)

    plt.title(f"Regression Error for {dataset_label}")
    plt.loglog(lambdas, errors[:,0], marker='.', label=f"Linear kernel")
    plt.loglog(lambdas, errors[:,1], marker='.', label=f"2nd order kernel")
    plt.loglog(lambdas, errors[:,2], marker='.', label=f"3rd order kernel")
    plt.loglog(lambdas, errors[:,3], marker='.', label=f"RBF kernel")
    plt.loglog(lambdas, errors[:,4], marker='.', label=f"Laplacian kernel")
    plt.xlabel(r"Regularization parameter $\lambda$")
    plt.ylabel("Mean Squared Error")
    if d_idx == 0: 
        hoi = True
        #plt.ylim((10**(0), 10**(1)))
    if d_idx == 1:
        hoi = True
        #plt.ylim((10**(0), 2*10**(1)))
    if d_idx == 2:
        hoi = True 
        #plt.ylim((10**(1), 10**(2)))
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()

    