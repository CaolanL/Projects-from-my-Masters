import numpy as np
import matplotlib.pyplot as plt

# For loading files and plotting only!
# Pick which files to compare, based off of tags below
q=5
cfl = 0.15 # Standard was 0.15
C_amd = 0.5
N = 2**q
grid_tag = f"grid2e{q}" # used for finding files
def clean(s):
    return str(s).replace('.', '')
Camd_tag = f"camd{clean(C_amd)}"
cfl_tag = f"cfl{clean(cfl)}"


folderAMD = f"results/AMD_{cfl_tag}_{Camd_tag}/{grid_tag}"
folderDNS = f"results/DNS_{cfl_tag}/{grid_tag}"

Energy_amd = np.loadtxt(f"{folderAMD}/energy_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat")
Energy_dns = np.loadtxt(f"{folderDNS}/energy_dns_{grid_tag}_{cfl_tag}.dat")

Diss_amd = np.loadtxt(f"{folderAMD}/dissipation_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat")
Diss_dns = np.loadtxt(f"{folderDNS}/dissipation_dns_{grid_tag}_{cfl_tag}.dat")

Spectra_amd = np.loadtxt(f"{folderAMD}/spectrum_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat")
Spectra_dns = np.loadtxt(f"{folderDNS}/spectrum_dns_{grid_tag}_{cfl_tag}.dat"
                         )
runtime_amd = np.loadtxt(f"{folderAMD}/runtime_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat")
runtime_dns = np.loadtxt(f"{folderDNS}/runtime_dns_{grid_tag}_{cfl_tag}.dat")

# 1. Kinetic energy:


t_amd, KE_amd = Energy_amd[:,0], Energy_amd[:,1]
t_dns, KE_dns = Energy_dns[:,0], Energy_dns[:,1]

plt.figure()
plt.plot(t_amd, KE_amd, label="AMD-LES")
plt.plot(t_dns, KE_dns, label="DNS")
plt.xlabel("Time")
plt.ylabel("Kinetic Energy")
plt.title(f"TGV: Kinetic Energy - Grid = {N}")
plt.grid()
plt.legend()
plt.tight_layout()

# 2. Dissipation:

t_eps_amd, eps_amd = Diss_amd[:,0], -Diss_amd[:,1]
t_eps_dns, eps_dns = Diss_dns[:,0], -Diss_dns[:,1]

plt.figure()
plt.plot(t_eps_amd, eps_amd, label="AMD-LES")
plt.plot(t_eps_dns, eps_dns, label="DNS")
plt.xlabel("Time")
plt.ylabel("Dissipation")
plt.title(f"TGV: Dissipation - Grid = {N}")
plt.grid()
plt.legend()
plt.tight_layout()

# 3. Energy spectra with k^{-5/3}:

# Load AMD spectrum
file_amd = f"results/AMD_{cfl_tag}_{Camd_tag}/{grid_tag}/spectrum_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat"
Spectra_amd = np.loadtxt(file_amd)
# Load DNS spectrum
file_dns = f"results/DNS_{cfl_tag}/{grid_tag}/spectrum_dns_{grid_tag}_{cfl_tag}.dat"
Spectra_dns = np.loadtxt(file_dns)
# Extract columns
k_amd, Ek_amd = Spectra_amd[:,0], Spectra_amd[:,1]
k_dns, Ek_dns = Spectra_dns[:,0], Spectra_dns[:,1]
# Remove k=0
maskA = k_amd > 0
maskD = k_dns > 0
kA, EA = k_amd[maskA], Ek_amd[maskA]
kD, ED = k_dns[maskD], Ek_dns[maskD]

plt.figure(figsize=(7,5))

# Plot spectra
plt.loglog(kA, EA, 'o-', markersize=3, label="AMD")
plt.loglog(kD, ED, 's-', markersize=3, label="DNS")

# Reference k^{-5/3}
kmin = max(kA[0], kD[0])
kmax = min(kA[-1], kD[-1])
k_ref = np.logspace(np.log10(kmin), np.log10(kmax), 200)

# Normalize in mid-spectrum
mid = len(kD)//3
C = ED[mid] * kD[mid]**(5/3)

plt.loglog(k_ref, C * k_ref**(-5/3), 'k--', label=r"$k^{-5/3}$")

plt.xlabel(r"$k$")
plt.ylabel(r"$E(k)$")
plt.title(f"TGV Energy Spectrum (N={N})")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()


plt.show()