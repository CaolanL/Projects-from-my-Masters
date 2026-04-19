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
lambd = 10**(-3)

nr_approximations = 30
rank_values = np.array(list(map(int, 10**np.linspace(0, 3, nr_approximations))))

test_rate = 0.3
nr_tests = round(test_rate*n)

# databases and kernels
datasets = [    
    #(data file,       name,      is_integer
    #(dataset_names[6], "Abalone", True),
    (dataset_names[1], "Pumadyn-8fm", False),
    (dataset_names[2], "Pumadyn-8nh", False)
]

kernels = [
    #kernel         , name           , plot color
    (linear_kernel(), "Linear kernel", "blue"),
    (second_order_kernel(), "2nd order kernel", "orange"),
    (third_order_kernel(), "3rd order kernel", "green"),
    (rbf_kernel(), "RBF kernel", "red"),
    (laplacian_kernel(), "Laplacian kernel", "purple")
]

# For each dataset, iterate over all kernels
for dataset_address, dataset_name, is_integer in datasets:
    
    min_value = 10**10
    max_value = 0

    for kernel, kernel_name, plot_color in kernels:
        print(f"started on {dataset_name} with {kernel_name}")

        # retrieve training data
        xs_training, ys_training, xs_testing, ys_testing = split_dataset_normalized(dataset_address, n, nr_tests)

        # build kernel matrix
        K = build_kernel_matrix(xs_training, kernel)

        # compute K accuracy
        K_y_predictor = construct_y_predictor(K, xs_training, ys_training, lambd, kernel)
        K_accuracy = mean_squared_error(xs_testing, ys_testing, K_y_predictor, is_integer)

        # compute SVD compositions
        Q, singular_values, Q_T = np.linalg.svd(K, hermitian=True)        

        # compute low-rank approximations
        approximation_accuracy = np.zeros(nr_approximations)
        for index, rank in enumerate(rank_values):

            # truncate the singular values and compute alpha
            D_inv = np.diag(1 / (singular_values[:rank] + lambd*n))
            alpha = Q[:,:rank] @ (D_inv @ (Q_T[:rank,:] @ ys_training))

            # determine the mean squared error
            y_predictor = lambda x : predict_y(xs_training, alpha, kernel, x)
            approximation_accuracy[index] = mean_squared_error(xs_testing, ys_testing, y_predictor, is_integer)

        # plot accuracy of K and low rank approximations
        plt.loglog(rank_values, K_accuracy*np.ones(nr_approximations), color=plot_color, ls="--")
        plt.loglog(rank_values, approximation_accuracy, label=f"{kernel_name}", color=plot_color, marker='.')

        min_value = min(min_value, min(approximation_accuracy))
        max_value = max(max_value, max(approximation_accuracy))

    # figure details
    plt.title(f"Low rank approximation accuracy - for {dataset_name}")
    plt.ylabel("Mean squared error")
    plt.xlabel("Rank of approximation")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.ylim(bottom = min_value/1.3, top = max_value*1.2)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
