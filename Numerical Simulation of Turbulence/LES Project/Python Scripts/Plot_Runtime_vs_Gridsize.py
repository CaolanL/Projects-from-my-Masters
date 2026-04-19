import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Loads files for all gridsizes, with cfl and C_amd below
cfl = 0.15
C_amd = 0.3

def clean(s):
    return str(s).replace('.', '')

cfl_tag = f"cfl{clean(cfl)}"
camd_tag = f"camd{clean(C_amd)}"

# Helper to extract q from filenames like "runtime_amd_grid2e5_camd01_cfl015.dat"  
def extract_q(filename):
    match = re.search(r"grid2e(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

# Find all AMD runtime files
amd_pattern = f"results/AMD_{cfl_tag}_{camd_tag}/grid2e*/runtime_amd_grid2e*_{camd_tag}_{cfl_tag}.dat"
amd_files = sorted(glob.glob(amd_pattern))

# Find all DNS runtime files
dns_pattern = f"results/DNS_{cfl_tag}/grid2e*/runtime_dns_grid2e*_{cfl_tag}.dat"
dns_files = sorted(glob.glob(dns_pattern))

grid_sizes_amd = []
runtimes_amd = []

grid_sizes_dns = []
runtimes_dns = []

# Process AMD files  
for f in amd_files:
    q = extract_q(f)
    if q is not None:
        N = 2**q
        runtime = np.loadtxt(f)
        grid_sizes_amd.append(N)
        runtimes_amd.append(runtime)

#  Process DNS files  
for f in dns_files:
    q = extract_q(f)
    if q is not None:
        N = 2**q
        runtime = np.loadtxt(f)
        grid_sizes_dns.append(N)
        runtimes_dns.append(runtime)

# Convert to arrays
grid_sizes_amd = np.array(grid_sizes_amd)
runtimes_amd = np.array(runtimes_amd)

grid_sizes_dns = np.array(grid_sizes_dns)
runtimes_dns = np.array(runtimes_dns)

# Sort by grid size  
idx_amd = np.argsort(grid_sizes_amd)
idx_dns = np.argsort(grid_sizes_dns)

grid_sizes_amd = grid_sizes_amd[idx_amd]
runtimes_amd = runtimes_amd[idx_amd]

grid_sizes_dns = grid_sizes_dns[idx_dns]
runtimes_dns = runtimes_dns[idx_dns]

# Plot  
plt.figure(figsize=(8,6))
plt.loglog(grid_sizes_amd, runtimes_amd, 'o-', label=f"AMD (C_amd={C_amd}, CFL={cfl})")
plt.loglog(grid_sizes_dns, runtimes_dns, 's-', label=f"DNS (CFL={cfl})")

plt.xlabel("Grid size N = 2^q")
plt.ylabel("Runtime [s]")
plt.title("Runtime Scaling: Grid Size vs Runtime")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

