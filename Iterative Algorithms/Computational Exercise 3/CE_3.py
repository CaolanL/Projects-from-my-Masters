import numpy as np
from numpy.linalg import svd
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import os



# 1. PARAMETERS

damage_ratio = 0.8  # % of pixels removed

gamma = 2.0 # step size
#Large γ -> Strong trust in know pixels, moves faster towards them 
#Small γ -> Less trust in known pixels, allows low-rank prior (the term with lambda) to dominate 

L = 0.8 #Lambda (regularization weight), has to be > tol=1e-5
#Large λ -> forces the reconstruction to be more low‑rank (can result in blurriness)
#Small λ -> reconstruction stays closer to the corrupted version (can be more noisy)

max_it = 100 # max iterations if convergence below the tolerance doesn't occur


# PROXIMAL OPERATORS


def prox_g(Y, Xcorr, mask, gamma):
    # Formula: prox_g(Y) = (Y + gamma * mask * Xcorr) / (1 + gamma * mask)
    # This works channel-wise automatically because mask is 3-channel.
    return (Y + gamma * mask * Xcorr) / (1 + gamma * mask)


def prox_f(Z, L, gamma):

    result = np.zeros_like(Z)

    # Process each color channel independently
    for c in range(3):
        # Compute SVD of the channel
        U, S, Vt = svd(Z[..., c], full_matrices=False)

        # Soft-threshold singular values
        S_new = np.maximum(S - gamma * L, 0)

        # Reconstruct the matrix
        result[..., c] = (U * S_new) @ Vt

    return result



# T-OPERATOR (THE MAIN ITERATION STEP)


def T(Y, Xcorr, mask, L, gamma):

    # First proximal step
    Pg = prox_g(Y, Xcorr, mask, gamma)

    # Compute the argument for prox_f
    correction_term = mask * (Pg - Xcorr)
    Z = 2 * Pg - Y - gamma * correction_term

    # Second proximal step
    Pf = prox_f(Z, L, gamma)

    # Combine everything
    return Y - Pg + Pf


#  MAIN INPAINTING LOOP
def inpaint_image(Xcorr, mask, L, gamma, max_it, tol=1e-5):


    Y = Xcorr.copy()  # Start from corrupted image

    for k in range(max_it):
        Y_new = T(Y, Xcorr, mask, L, gamma)

        # Compute relative change
        diff = np.linalg.norm(Y_new - Y) / (np.linalg.norm(Y) + 1e-12)

        print(f"Iteration {k+1}, change = {diff:.6f}")

        if diff < tol:
            print("Converged.")
            break

        Y = Y_new

    return Y



# APPLICATION TO IMAGE


# Load image (must be RGB)
X = img_as_float(io.imread("Icarus_512.png")) #512x512 resolution
if X.shape[2] == 4: 
    X = X[..., :3] # Keep only RGB channels - removes alpha channel, used for opacity in images

# Create a random mask (simulate damaged pixels)
np.random.seed(42)
mask_2d = (np.random.rand(*X.shape[:2]) > damage_ratio).astype(float)

# Expand mask to 3 channels
mask = np.repeat(mask_2d[:, :, None], 3, axis=2)

# Apply corruption
Xcorr = mask * X

# Run inpainting
Xrec = inpaint_image(Xcorr, mask, L, gamma, max_it)



# PLOTS

# Extract image filename
image_filename = "Icarus_512.png" 
image_name = os.path.basename(image_filename)

# Display results
plt.figure(figsize=(12, 4))
plt.suptitle(f"Results for {image_name},damage={damage_ratio:.2f}, γ={gamma},  λ={L} ", fontsize=14)

# Subplot 1: Original
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(X)
ax1.set_title("Original", fontsize=12)
ax1.axis('off')
# Subplot 2: Corrupted
ax2 = plt.subplot(1, 3, 2)
ax2.imshow(Xcorr)
ax2.set_title("Corrupted", fontsize=12)
ax2.axis('off')
# Subplot 3: Recovered
ax3 = plt.subplot(1, 3, 3)
ax3.imshow(np.clip(Xrec, 0, 1))
ax3.set_title("Recovered", fontsize=12)
ax3.axis('off')

plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.85)  # leave space for main title
plt.show()

