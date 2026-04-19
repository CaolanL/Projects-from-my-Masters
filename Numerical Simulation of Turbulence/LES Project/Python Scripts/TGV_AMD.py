import numpy as np
import time
from EnergySpectrum_module import energy_spectrum
import os

# Domain
lx, ly, lz = 2*np.pi, 2*np.pi, 2*np.pi


# Grid
q=5
nx, ny, nz = 2**q, 2**q, 2**q
dx, dy, dz = lx/nx, ly/ny, lz/nz


# Cell centers
x = np.linspace(dx/2, lx - dx/2, nx)
y = np.linspace(dy/2, ly - dy/2, ny)
z = np.linspace(dz/2, lz - dz/2, nz)

# AMD constant
C_amd = 0.3 # Usually ~ 0.1 to 0.3

# Time step
cfl = 0.15 # 0.1 - 0.3 or so (0.15 original)
dt = cfl*dx
lt = 20
nt = round(lt/dt)



# Reynolds number
Re = 1600   


# Initial velocity
u = np.zeros((nx,ny,nz))
v = np.zeros((nx,ny,nz))
w = np.zeros((nx,ny,nz))



# Initial velocities (now vectorised)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u =  np.cos(X)*np.sin(Y)*np.sin(Z)
v = -np.sin(X)*np.cos(Y)*np.sin(Z)
w =  np.zeros_like(u)

# Initialization of pressure
p = np.zeros((nx,ny,nz))


# Initialization kinetic energy and dissipation
e_kin = np.zeros((nt+1,2))
dissp = np.zeros((nt  ,2))


# Cell-to-face interpolation
# velocities and fluxes evaluated at faces, not cell centres
def cell2face(f,ax):
   return 0.5*(f + np.roll(f, 1, axis=ax))


def meanx(f):
    return cell2face(f,0)
def meany(f):
    return cell2face(f,1)
def meanz(f):
    return cell2face(f,2)


# Finite Differences

# Backward difference for faces
# for computing gradients of viscous fluxes at faces
def dfdx_face(f):
    return (f - np.roll(f, 1, axis=0))/dx
def dfdy_face(f):
    return (f - np.roll(f, 1, axis=1))/dy
def dfdz_face(f):
    return (f - np.roll(f, 1, axis=2))/dz

# Forward difference for cells
# for computing divergence of fluxes at cell centres
def dfdx_cell(f):
    return (np.roll(f, -1, axis=0) - f)/dx
def dfdy_cell(f):
    return (np.roll(f, -1, axis=1) - f)/dy
def dfdz_cell(f):
    return (np.roll(f, -1, axis=2) - f)/dz

# Second order central
# Used only for AMD
def dfdx_centered(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)

def dfdy_centered(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dy)

def dfdz_centered(f):
    return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2*dz)


# AMD Implementation
def compute_nu_amd_using_gradients(
    du_dx, du_dy, du_dz,
    dv_dx, dv_dy, dv_dz,
    dw_dx, dw_dy, dw_dz
):
    # Gradient tensors
    G11, G12, G13 = du_dx, du_dy, du_dz 
    G21, G22, G23 = dv_dx, dv_dy, dv_dz 
    G31, G32, G33 = dw_dx, dw_dy, dw_dz
    #Symmetric strain-rate tensors
    S11 = G11
    S22 = G22 
    S33 = G33 
    S12 = 0.5*(G12 + G21) 
    S13 = 0.5*(G13 + G31) 
    S23 = 0.5*(G23 + G32) 
     
    numer = (
        G11*S11 + G22*S22 + G33*S33 +
        G12*S12 + G13*S13 + G23*S23
    )
    denom = ( 
        G11**2 + G12**2 + G13**2 + 
        G21**2 + G22**2 + G23**2 + 
        G31**2 + G32**2 + G33**2
    )
    DeltaSquared = dx*dx #filter width set equal to grid cell width
    nu_amd = np.maximum(0.0, (-C_amd *DeltaSquared* numer) / (denom ))
    return nu_amd, numer, denom

# Convective velocities
def convection(u,v,w):
    return meanx(u), meany(v), meanz(w)

# Flux with AMD viscosity
def flux(u, nu_amd):
    nu_x = 1/Re + meanx(nu_amd)
    nu_y = 1/Re + meany(nu_amd)
    nu_z = 1/Re + meanz(nu_amd)

    fx = u_convect * meanx(u) - nu_x * dfdx_face(u)
    fy = v_convect * meany(u) - nu_y * dfdy_face(u)
    fz = w_convect * meanz(u) - nu_z * dfdz_face(u)
    return fx, fy, fz


# Discrete divergence at cell centre
def divergence(fx,fy,fz):
    return ( dfdx_cell(fx) + dfdy_cell(fy) + dfdz_cell(fz) )




# Incompressibility constraint

# Wave numbers FFT (multiplied by 2*pi)
kx = np.fft.fftfreq(nx, d=dx)*2*np.pi
ky = np.fft.fftfreq(ny, d=dy)*2*np.pi
kz = np.fft.fftfreq(nz, d=dz)*2*np.pi
# Modified wave numbers for pressure Poisson equation
km = np.zeros((nx, ny, nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            km[i,j,k] =            2*(np.cos(kx[i]*dx)-1)/(dx*dx)
            km[i,j,k] = km[i,j,k]+ 2*(np.cos(ky[j]*dy)-1)/(dy*dy)
            km[i,j,k] = km[i,j,k]+ 2*(np.cos(kz[k]*dz)-1)/(dz*dz)
km[0,0,0] = 1  # avoid division by zero


# FFT solver for pressure Poisson equation
def pressure(div):
    p_hat = np.fft.fftn(div) / km
    p_hat[0, 0, 0] = 0  # zero mean pressure
    return np.real(np.fft.ifftn(p_hat))




def energy(u,v,w):   
    return np.sum(u**2 + v**2 + w**2)/(2*nx*ny*nz)
 

t = 0


# Initial convective velocity
u_convect,v_convect,w_convect = convection(u,v,w)


# Initial kinetic energy
e_kin[0,0] = t
e_kin[0,1] = energy(u,v,w)




# TIME LOOP

start_time = time.time() 
for n in range(nt):
    
    #Second-order central 
    du_dx = dfdx_centered(u); du_dy = dfdy_centered(u); du_dz = dfdz_centered(u) 
    dv_dx = dfdx_centered(v); dv_dy = dfdy_centered(v); dv_dz = dfdz_centered(v) 
    dw_dx = dfdx_centered(w); dw_dy = dfdy_centered(w); dw_dz = dfdz_centered(w) 

    # AMD viscosity
    nu_amd, numer, denom = compute_nu_amd_using_gradients(
        du_dx, du_dy, du_dz,
        dv_dx, dv_dy, dv_dz,
        dw_dx, dw_dy, dw_dz
    )

    # Make AMD viscosity only act from the inertial subrange onwards
    # if t < 7:
    #     nu_amd = np.zeros_like(nu_amd)

    fx,fy,fz = flux(u, nu_amd)
    u -= dt*divergence(fx,fy,fz)

    fx,fy,fz = flux(v, nu_amd)
    v -= dt*divergence(fx,fy,fz)

    fx,fy,fz = flux(w, nu_amd)
    w -= dt*divergence(fx,fy,fz)

    u_convect, v_convect, w_convect = convection(u,v,w)
    
    # Solve pressure Poisson eq
    p = pressure(divergence(u_convect,v_convect,w_convect))

    # Subtract pressure gradient
    u_convect -=  dfdx_face(p)
    v_convect -=  dfdy_face(p)
    w_convect -=  dfdz_face(p)


    # Update velocities at cell center
    u -= dfdx_cell(meanx(p))
    v -= dfdy_cell(meany(p))
    w -= dfdz_cell(meanz(p))
                            
    # Update time
    t = t + dt

    # Kinetic energy
    e_kin[n+1,0] = t
    e_kin[n+1,1] = energy(u,v,w)

    print(t,e_kin[n+1,1])

    # Diagnostic print
    # if n % 10 == 0:
    #     print("AMD_num min/max:", np.min(numer), np.max(numer))
    # if n % 50 == 0:
    #     print("denom min/max:", np.min(denom), np.max(denom))

    # if n % 10 == 0:
    #     print("nu_amd min/max:", np.min(nu_amd), np.max(nu_amd))



    # Energy dissipation
    dissp[n,0] = t-0.5*dt
    dissp[n,1] = (e_kin[n+1,1]-e_kin[n,1])/dt


end_time = time.time()
runtime = end_time - start_time
print("Runtime [s] =", runtime)

k_amd, Ek_amd = energy_spectrum(u, v, w, lx, ly, lz) #requires EnergySpectrum_module.py

# Tags for naming files and their corresponding folders
grid_tag = f"grid2e{q}"
def clean(s):
    return str(s).replace('.', '')
Camd_tag = f"camd{clean(C_amd)}"
cfl_tag = f"cfl{clean(cfl)}"

# Create folders if doesn't already exist
folderAMD = f"results/AMD_{cfl_tag}_{Camd_tag}/{grid_tag}"
os.makedirs(folderAMD, exist_ok=True)

np.savetxt(f"{folderAMD}/runtime_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat", np.array([runtime]))
np.savetxt(f"{folderAMD}/energy_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat",e_kin,fmt='%.12f')
np.savetxt(f"{folderAMD}/dissipation_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat",dissp,fmt='%.12f')
np.savetxt(f"{folderAMD}/spectrum_amd_{grid_tag}_{Camd_tag}_{cfl_tag}.dat", np.column_stack((k_amd, Ek_amd)))