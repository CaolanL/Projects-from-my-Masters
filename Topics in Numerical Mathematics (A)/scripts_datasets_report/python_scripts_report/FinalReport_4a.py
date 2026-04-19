# setup
import matplotlib.pyplot as plt
import numpy as np

from functions import * 
from kernels import *

rng = np.random.default_rng(42)   #fix the randomness seed for reproducability

dataset_names = [
    "pumadyn-8fh.data",     #0
    "pumadyn-8fm.data",     #1
    "pumadyn-8nh.data",     #2
    "pumadyn-8nm.data",     #3
    "gas_sensor_full.data", #4
    "wine_full_binary.data",#5
    "abalone_clean.data",   #6
    "bank-8fm.data",        #7
    "BostonHousing.data",   #8
    "housing.data",         #9
    "uniform_random.data",  #10
    "normal_distributed.data", #11
    "max_distributed.data"     #12
]


# parameters
n = 2000
c = 0.1

# databases and kernels
datasets = [    
    #(data file,       name,          kernel, kernel_name
    #(dataset_names[12], "Max distribution")
    (dataset_names[11], "Normal distribution"),
    (dataset_names[10], "Uniform distribution")
    #(dataset_names[6], "Abalone"),
    #(dataset_names[1], "Pumadyn-8fm"),
    #(dataset_names[2], "Pumadyn-8nh")
]

kernels = [
    (linear_kernel(), "Linear kernel"),
    (second_order_kernel(), "2nd order kernel"),
    (third_order_kernel(), "3rd order kernel"),
    (rbf_kernel(), "RBF kernel"),
    (laplacian_kernel(), "Laplacian kernel")
]

for dataset_address, dataset_name in datasets:
    for kernel, kernel_name in kernels:
        print(f"started on {dataset_name} with {kernel_name}")

        # retrieve training data
        xs_training, _, _, _ = split_dataset_normalized(dataset_address, n, 1)

        # build kernel matrix
        K = build_kernel_matrix(xs_training, kernel)

        # compute SVD compositions
        singular_values = np.linalg.svd(K, compute_uv=False, hermitian=True)

        # plot singular values
        plt.loglog(range(n), singular_values, label=f"{kernel_name}")

    plt.title(r"Ordered singular values of kernel matrix - for %s with n=%d" % (dataset_name, n))
    plt.ylabel(r"Singular value $\sigma_i$")
    plt.xlabel("Index i")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,1,0.95])     
    plt.show()