import numpy as np


def energy_spectrum(u, v, w, lx, ly, lz, nbins=None):
    nx, ny, nz = u.shape
    if nbins is None:
        nbins = nx // 2 #can change bins
    # Fourier transforms
    u_hat = np.fft.fftn(u)
    v_hat = np.fft.fftn(v)
    w_hat = np.fft.fftn(w)

    # Wavenumbers
    kx = np.fft.fftfreq(nx, d=lx/nx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=ly/ny) * 2*np.pi
    kz = np.fft.fftfreq(nz, d=lz/nz) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Spectral energy density
    E_k = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2) / (nx*ny*nz)**2

    k_flat = k_mag.ravel()
    E_flat = E_k.ravel()

    k_max = k_flat.max()
    k_bins = np.linspace(0.0, k_max, nbins+1)
    k_center = 0.5*(k_bins[:-1] + k_bins[1:])

    E_shell = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=int)

    idx = np.digitize(k_flat, k_bins) - 1
    valid = (idx >= 0) & (idx < nbins)
    for i, e in zip(idx[valid], E_flat[valid]):
        E_shell[i] += e
        counts[i] += 1

    nonzero = counts > 0
    E_shell[nonzero] /= counts[nonzero]

    return k_center[nonzero], E_shell[nonzero]
