import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kernels import linear_kernel_evaluation

#eps_machine = np.finfo(float).eps * 10**(4) #10^4 just to be sure, Cholesky sometimes failed/acted weirdly
eps_stab = 10**(-4)

# ==========================================================================================
# Data set related
# ==========================================================================================

# retrieve the sample rows of xs and the values of y
def retrieve_xs_and_ys(dataset, remove_1st_column = False):
    if remove_1st_column == True:
        xs = dataset[:,1:-1]       
    else:  
        xs = dataset[:,:-1]
    ys = dataset[:,-1]        
    return xs, ys

# retrieve subsets for testing and training
def split_xs_and_ys(xs, ys, train_size=2000, test_size=100, rng=np.random.default_rng(42)):
    indices = rng.choice(len(xs), size = train_size + test_size, replace = False)
    xs_subset = xs[indices,:]
    ys_subset = ys[indices]
    xs_training, xs_testing, ys_training, ys_testing = train_test_split(
       xs_subset, 
       ys_subset, 
       test_size = test_size, 
       random_state=42
    )
    
    return xs_training, ys_training, xs_testing, ys_testing

def split_dataset(dataset_name, train_size=2000, test_size=100, rng=np.random.default_rng(42)):
    
    dataset = np.loadtxt(f"./datasets/{dataset_name}", delimiter=None)
    xs, ys = retrieve_xs_and_ys(dataset)

    xs_training, ys_training, xs_testing, ys_testing = split_xs_and_ys(xs, ys, train_size, test_size, rng)

    return xs_training, ys_training, xs_testing, ys_testing

def split_dataset_normalized(dataset_name, train_size=2000, test_size=100, rng=np.random.default_rng(42), m = 0):
    
    dataset = np.loadtxt(f"./datasets/{dataset_name}", delimiter=None)
    no_duplicates_dataset = remove_duplicates(dataset)
    if dataset_name == "housing.data":
        xs, ys = retrieve_xs_and_ys(no_duplicates_dataset, remove_1st_column = True)
    else: 
        xs, ys = retrieve_xs_and_ys(no_duplicates_dataset)    
    if m > 0:
        m_goal = m
        n, m_curr = xs.shape    
        xs_new = np.zeros((n,m_goal))
     
        full_copies = int(np.floor( m_goal / m_curr ))
        for i in range(full_copies):
            xs_new[:,i*m_curr:(i+1)*m_curr] = xs
        end_idx = (full_copies)*m_curr
        partial = m_goal % m_curr
        for i in range(partial):
            xs_new[:,end_idx+i] = xs[:,i]
        xs = xs_new

    xs_training, ys_training, xs_testing, ys_testing = split_xs_and_ys(xs, ys, train_size, test_size, rng)

    #normalize the data set: https://www.geeksforgeeks.org/machine-learning/understanding-kernel-ridge-regression-with-sklearn/
    scaler = MinMaxScaler()
    xs_training = scaler.fit_transform(xs_training)
    xs_testing = scaler.transform(xs_testing)
    
    return xs_training, ys_training, xs_testing, ys_testing

def remove_duplicates(dataset):   
    xs = dataset[:, :-1]
    _, indices = np.unique(xs, axis=0, return_index=True)   #returns indices of the unique rows
    dataset = dataset[indices]      

    return dataset


# ==========================================================================================
# KRR related
# ==========================================================================================

def build_kernel_matrix(xs_training, kernel=linear_kernel_evaluation):
    n = len(xs_training)
    K = np.zeros([n,n])
    for i in range(n):    #compute only half of the matrix and copy it to the other half
        for j in range(i,n):
            entry = kernel(xs_training[i], xs_training[j])
            K[i,j] = entry 
            K[j,i] = entry
    return K

def construct_y_predictor(K, xs_training, ys_training, lambd=10**(-3), kernel=linear_kernel_evaluation):
    n = len(ys_training)

    alpha_hat = sp.linalg.solve(
       K + n * lambd * np.identity(n), 
       ys_training, 
       assume_a='symmetric',    # using only symmetry
       check_finite = False
    ) 
    
    y_predictor = lambda x : predict_y(xs_training, alpha_hat, kernel, x)

    return y_predictor

# Evaluate f at point x, using datapoints x_i: f(x) = sum_i (alpha_i * k(x,x_i))
def predict_y(xs_training, alpha_hat, kernel, x):
    n = len(xs_training)

    y_pred = 0
    for i in range(n):
        y_pred += alpha_hat[i] * kernel(x, xs_training[i])

    return y_pred   
    
def construct_C_and_W(xs_training, kernel, columns_indices, n, s):
    C = np.zeros((n,s))  
    W = np.zeros((s,s)) 

    for row_nr, row_idx in enumerate(columns_indices): 
        for col_nr, col_idx in enumerate(columns_indices[:row_nr+1]):
            entry = kernel(xs_training[row_idx], xs_training[col_idx])
            W[row_nr, col_nr] = entry 
            W[col_nr, row_nr] = entry
    
    for col_nr, col_idx in enumerate(columns_indices):    
        C[:,col_nr] = [ kernel(xs_training[row_idx], xs_training[col_idx]) for row_idx in range(n) ]

    return C, W

def find_Khat(C, W, columns_indices):
    n, s = C.shape
    eps_Omega = np.zeros((n,s))
    for j, i in enumerate(columns_indices):
        eps_Omega[i,j] = eps_stab
    Omega_eps_Omega = eps_stab * np.eye(s)

    Y = C + eps_Omega
    H = np.linalg.cholesky(W + Omega_eps_Omega, upper = True)
    Z = sp.linalg.solve_triangular(H, Y.T, trans = 1, lower = False)
    U, Sigma, _ = np.linalg.svd(Z.T, full_matrices = False) #We only need the first s columns of U, so no full matrices
    Lambda = np.maximum(0, Sigma**2 - eps_stab)    

    return U @ (np.diag(Lambda) @ U.T)

# ==========================================================================================
# Error analysis
# ==========================================================================================

def relative_error(y_true, y_pred):
    if y_true < 10**(-6):
        return abs(y_true - y_pred) / abs(y_pred)
    else: 
        return abs(y_true - y_pred) / abs(y_true)
    
def mean_squared_relative_error(xs_testing, ys_testing, y_predictor):
    num_tests = len(ys_testing)

    mse = 0.0
    for (x_test, y_test) in zip(xs_testing, ys_testing):
        mse += (1.0 / num_tests) * relative_error(y_test, y_predictor(x_test))**2
    
    return mse

def mean_squared_error(xs_testing, ys_testing, y_predictor, integer = False):
    num_tests = len(ys_testing)

    mse = 0.0
    for (x_test, y_test) in zip(xs_testing, ys_testing):
        if integer == True:
            mse += (1.0 / num_tests) * abs(y_test - round(y_predictor(x_test)))**2
        else: 
            mse += (1.0 / num_tests) * abs(y_test - y_predictor(x_test))**2
    
    return mse


def accuracy(xs_testing, ys_testing, y_predictor):
    num_tests = len(ys_testing)
    successes = 0.0
    for (x_test, y_test) in zip(xs_testing, ys_testing):
        y_target = round(y_test)
        y_pred = round(y_predictor(x_test))
        if abs(y_target - y_pred) < 0.5:
            print(f"succes")
            successes += 1
    return successes / num_tests

# ==========================================================================================
# Column selection
# ==========================================================================================

def probabilities_uniform(K, p, lambd, rng):
    n = len(K)
    scores = np.ones(n)
    prob_distribution = scores/np.sum(scores)
    return prob_distribution

def probabilities_diagonal(K, p, lambd, rng):
    n = len(K)
    scores = np.zeros(n)
    for i in range(n):
        scores[i]= K[i,i]    
    prob_distribution = scores/np.sum(scores)
    return prob_distribution


def probabilities_exact_leverage(K, p, lambd, rng):
    #the rank k in the leverage score definition, is choosen the same as the sampling parameter p in the approximate LR leverage
    k = p           
    
    # Find the decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.array(eigenvalues)
    sorted_index = eigenvalues.argsort()[::-1]
    U = eigenvectors[:, sorted_index]

    U_1 = eigenvectors[:, :k]
    scores = np.sum(U_1**2, axis = 1)
    prob_distribution = scores/np.sum(scores)
    return prob_distribution

def probabilities_exact_lambda_ridge(K, p, lambd, rng):
    n = len(K)
    Sigma, U = np.linalg.eigh(K)
    scores = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            scores[i] += (Sigma[j])/(Sigma[j]+n*lambd) * U[i,j]**2

    prob_distribution = scores/np.sum(scores)
    return prob_distribution

def probabilities_approx_lambda_ridge(K, p, lambd, rng):
    # p = sampling parameter
    # Note that this method does NOT compute all of K!
    n = len(K)
    diag_K = np.array([K[i,i] for i in range(n)])
    probs = diag_K / np.sum(diag_K)
    indices = rng.choice(n, size=p, replace=False, p=probs) # Indices of p sampled columns of K

    C = np.zeros((n,p))  
    W = np.zeros((p,p)) 
    for j, k in enumerate(indices):
        C[:,j] = K[:,k]

    for j, k in enumerate(indices):
        for a, b in enumerate(indices):
            W[j,a] = K[k,b] # W = K[sampling_indices, sampling_indices]

    # Construct B such that B B^T = C W^{-1} C^T
    # Using Cholesky: W = R^T R  => W^{-1} = R^{-1} R^{-T}
    # Then B B^T = (C R^{-1}) (R^{-T} C^T)
    # Then B = C R^{-1}
    W_reg = W + eps_stab*np.eye(p) #Added for stability
    R = np.linalg.cholesky(W_reg)
    R_inv = np.linalg.inv(R)
    B = C @ R_inv

    M = np.linalg.inv(B.T @ B + n*lambd*np.eye(p))
    LR_leverage_scores = []
    for i in range(n):
        Bi = B[i,:]
        l_i = Bi @ M @ Bi.T
        LR_leverage_scores.append(l_i)
    LR_leverage_scores = np.array(LR_leverage_scores)

    prob_distribution = LR_leverage_scores / np.sum(LR_leverage_scores)
    return prob_distribution

# ===================================================================
# Specific efficient/optimized functions for proper run-time analysis
# ===================================================================

def efficient_probabilities_uniform(X, kernel, p, lambd, rng):
    n = X.shape[0]
    return np.ones(n) / n

def efficient_probabilities_diagonal(X, kernel, p, lambd, rng):
    n = X.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        scores[i]= np.abs(kernel(X[i,:],X[i,:]))      
    prob_distribution = scores/np.sum(scores)
    return prob_distribution

def efficient_probabilities_approx_lambda_ridge(X, kernel, p, lambd, rng):
    # p = sampling parameter
    #Note that this method does NOT compute all of K!
    n = X.shape[0] # X is  n by m
    diag_K = np.array([kernel(X[i],X[i]) for i in range(n)])
    probs = diag_K / np.sum(diag_K)
    indices = rng.choice(n, size=p, replace=False, p=probs) # Indices of p*n sampled columns of K

    C, W = construct_C_and_W(X, kernel, indices, n, p)
     
    # Construct B such that B B^T = C W^{-1} C^T
    # Using Cholesky: W = R^T R  => W^{-1} = R^{-1} R^{-T}
    # Then B B^T = (C R^{-1}) (R^{-T} C^T)
    # Then B = C R^{-1} (i.e. B @ R = C or R.T @ B.T = C.T)
    W_reg = W + eps_stab * np.eye(p) #Added for stability
    R = np.linalg.cholesky(W_reg)
    B_tranpose = sp.linalg.solve_triangular(R, C.T, trans=1, check_finite=False) 

    # compute leverage scores
    M = np.linalg.inv(B_tranpose @ B_tranpose.T + n*lambd*np.eye(p))
    LR_leverage_scores = []
    for i in range(n):
        Bi = B_tranpose.T[i,:]
        l_i = Bi @ M @ Bi.T
        LR_leverage_scores.append(l_i)
    LR_leverage_scores = np.array(LR_leverage_scores)
    LR_leverage_scores = np.maximum(LR_leverage_scores, 0)
    
    prob_distribution = LR_leverage_scores / np.sum(LR_leverage_scores)
    return prob_distribution


#not really efficient, but format needed for testing
def efficient_probabilities_exact_leverage(X, kernel, p, lambd, rng):
    #the rank k in the leverage score definition, is choosen to be equal to sthe same as the sampling parameter p in the approximate LR leverage
    k = 10*p         #manually fixed for simplicity  
    
    K = build_kernel_matrix(X, kernel)
    # Find the decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.array(eigenvalues)
    sorted_index = eigenvalues.argsort()[::-1]
    U = eigenvectors[:, sorted_index]

    U_1 = U[:, :k]
    scores = np.sum(U_1**2, axis = 1)
    prob_distribution = scores/np.sum(scores)
    return prob_distribution

#not really efficient, but format needed for testing
def efficient_probabilities_exact_lambda_ridge(X, kernel, p, lambd, rng):
    K = build_kernel_matrix(X, kernel)
    n = len(K)
    Sigma, U = np.linalg.eigh(K)
    scores = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            scores[i] += (Sigma[j])/(Sigma[j]+n*lambd) * U[i,j]**2

    prob_distribution = scores/np.sum(scores)
    return prob_distribution


def efficient_find_Khat(C, W, columns_indices):
    n, s = C.shape

    eps_Omega = np.zeros((n,s))
    for j, i in enumerate(columns_indices):
        eps_Omega[i,j] = eps_stab
    Omega_eps_Omega = eps_stab * np.eye(s)

    Y = C + eps_Omega
    H = np.linalg.cholesky(W + Omega_eps_Omega, upper = True)
    Z_transpose = sp.linalg.solve_triangular(H, Y.T, trans = 1, lower = False) # solving H^T @ Z^T = Y^T
    U, Sigma, _ = np.linalg.svd(Z_transpose.T, full_matrices = False) #We only need the first s columns of U, so no full matrices
    Lambda = np.maximum(0, Sigma**2 - eps_stab)    

    return U, Lambda

def efficient_construct_y_predictor(U, Lambda, xs_training, ys_training, lambd, kernel):
    n = len(ys_training)
    s = len(Lambda)
    d = 1/(lambd*n)
    Dy = d * ys_training
    V = U.T
    inv = 1/(1/Lambda + d)
    invVBy = np.multiply(inv, V @ Dy)
    alpha_hat = Dy - d*U @ invVBy  

    y_predictor = lambda x : predict_y(xs_training, alpha_hat, kernel, x)

    return y_predictor
