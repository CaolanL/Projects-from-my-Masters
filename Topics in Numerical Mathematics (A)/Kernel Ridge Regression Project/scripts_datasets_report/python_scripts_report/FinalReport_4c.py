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
c = 0.1

# databases and kernels
datasets = [    
    #(data file,       name,          kernel, kernel_name
    #(dataset_names[12], "Max distribution")
    #(dataset_names[11], "Normal distribution"),
    #(dataset_names[10], "Uniform distribution")
    #(dataset_names[6], "Abalone")
    #(dataset_names[1], "Pumadyn-8fm"),
    (dataset_names[2], "Pumadyn-8nh")
]

kernels = [
    (linear_kernel(), "Linear kernel"),
    (second_order_kernel(c), "2nd order kernel"),
    (third_order_kernel(c), "3rd order kernel"),
    (rbf_kernel(c), "RBF kernel"),
    (laplacian_kernel(c), "Laplacian kernel")
]

drop_value = 1e-6
ns = np.array([20,40,100,200,400,1000,2000,4000])

for dataset_address, dataset_name in datasets:
    for kernel, kernel_name in kernels:
        print(f"started on {dataset_name} with {kernel_name}")

        drop_indices = np.zeros(len(ns))
        for n_index, n in enumerate(ns):
            # retrieve training data
            xs_training, _, _, _ = split_dataset_normalized(dataset_address, n, 1)

            # build kernel matrix
            K = build_kernel_matrix(xs_training, kernel)

            # compute SVD compositions
            singular_values = np.linalg.svd(K, compute_uv=False, hermitian=True)

            after_drop = [ idx for idx in range(n) if singular_values[idx] <= drop_value ]
            drop_index = (after_drop[0]) if len(after_drop)!=0 else -1
            drop_indices[n_index] = drop_index

        # plot singular values
        indices = [ idx for idx in range(len(ns)) if (drop_indices[idx] != -1) ]
        plt.loglog(ns[indices], drop_indices[indices], label=f"{kernel_name}", marker=".")

    plt.title(r"Drop-off for singular values when varying n - for %s" % (dataset_name))
    plt.ylabel(r"Index of first singular value smaller than (1e-6)")
    plt.xlabel("n")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.show()