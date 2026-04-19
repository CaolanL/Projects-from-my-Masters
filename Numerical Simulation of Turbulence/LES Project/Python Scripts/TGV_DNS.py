import numpy as np
import time
from EnergySpectrum_module import energy_spectrum
import os
# Domain
lx, ly, lz = 2*np.pi, 2*np.pi, 2*np.pi


# Grid
q=6
nx, ny, nz = 2**q, 2**q, 2**q
dx, dy, dz = lx/nx, ly/ny, lz/nz

# Cell centers
x = np.linspace(dx/2, lx - dx/2, nx)
y = np.linspace(dy/2, ly - dy/2, ny)
z = np.linspace(dz/2, lz - dz/2, nz)


# Faces
x_f = np.linspace(0,lx,nx+1)
y_f = np.linspace(0,ly,ny+1)
z_f = np.linspace(0,lz,nz+1)


# Time step
cfl = 0.15
dt = cfl*dx
lt = 20
nt = round(lt/dt)



# Reynolds number
re = 1600   


# Initial velocity at cell centers
u = np.zeros((nx,ny,nz))
v = np.zeros((nx,ny,nz))
w = np.zeros((nx,ny,nz))


# The initial velocity is given by
# u(x,y,z) = cos(x)*sin(y)*sin(z)
# v(x,y,z) =-sin(x)*cos(y)*sin(z)
# w(x,y,z) = 0.0
# Cos and sin at cell centers
cos = np.zeros(nx)
sin = np.zeros(nx)
for i in range(nx):
    cos[i] = np.cos(x[i])
    sin[i] = np.sin(x[i])
# Initial velocity at cell centers
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            u[i,j,k] = cos[i]*sin[j]*sin[k]
            v[i,j,k] =-sin[i]*cos[j]*sin[k]
            w[i,j,k] = 0.0


  
# Initialization of convective velocities
u_convect = np.zeros((nx,ny,nz))
v_convect = np.zeros((nx,ny,nz))
w_convect = np.zeros((nx,ny,nz))


# Initialization of fluxes
fx = np.zeros((nx,ny,nz))
fy = np.zeros((nx,ny,nz))
fz = np.zeros((nx,ny,nz))


# Initialization of pressure
p = np.zeros((nx,ny,nz))


# Initialization kinetic energy and dissipation
e_kin = np.zeros((nt+1,2))
dissp = np.zeros((nt  ,2))




# Cell-to-face interpolation

def cell2face(f,ax):
   return (f + np.roll(f, 1, axis=ax))/2


def meanx(f):
    return cell2face(f,0)
def meany(f):
    return cell2face(f,1)
def meanz(f):
    return cell2face(f,2)



# Finite Differences

def dfdx_face(f):
    return (f - np.roll(f, 1, axis=0))/dx
def dfdy_face(f):
    return (f - np.roll(f, 1, axis=1))/dy
def dfdz_face(f):
    return (f - np.roll(f, 1, axis=2))/dz


def dfdx_cell(f):
    return (np.roll(f, -1, axis=0) - f)/dx
def dfdy_cell(f):
    return (np.roll(f, -1, axis=1) - f)/dy
def dfdz_cell(f):
    return (np.roll(f, -1, axis=2) - f)/dz



# Convective and viscous flux at faces

def flux(u):
    fx = u_convect*meanx(u) - dfdx_face(u)/re
    fy = v_convect*meany(u) - dfdy_face(u)/re
    fz = w_convect*meanz(u) - dfdz_face(u)/re
    return(fx,fy,fz)



# Face-normal velocity from cell-to-face interpolation

def convect(u,v,w):
    u_convect = meanx(u)
    v_convect = meany(v)
    w_convect = meanz(w)
    return (u_convect,v_convect,w_convect)



# Discrete divergence at cell center

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
    e_kin = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                e_kin = e_kin + (u[i,j,k]**2) + (v[i,j,k]**2) + (w[i,j,k]**2)   
    return e_kin/(2*nx*ny*nz)
 

       

t = 0


# Initial convective velocity
u_convect,v_convect,w_convect = convect(u,v,w)


# Initial kinetic energy
e_kin[0,0] = t
e_kin[0,1] = energy(u,v,w)




# Start of time-stepping

start_time = time.time() 
for n in range(nt):
    

    # Tentative velocity
    
    fx,fy,fz = flux(u)
    u = u - dt*divergence(fx,fy,fz)
    
    fx,fy,fz = flux(v)
    v = v - dt*divergence(fx,fy,fz)
    
    fx,fy,fz = flux(w)
    w = w - dt*divergence(fx,fy,fz) 
    

    # Tentative convective velocity
    u_convect, v_convect, w_convect = convect(u,v,w)
  

    # Solve pressure Poisson eq
    p = pressure(divergence(u_convect,v_convect,w_convect))


    # Add pressure gradient to face-normal velocity
    u_convect = u_convect - dfdx_face(p)
    v_convect = v_convect - dfdy_face(p)
    w_convect = w_convect - dfdz_face(p)


    # Update velocity at cell center
    u = u - dfdx_cell(meanx(p))
    v = v - dfdy_cell(meany(p))
    w = w - dfdz_cell(meanz(p))
                            

    # Update time
    t = t + dt

    # Kinetic energy
    e_kin[n+1,0] = t
    e_kin[n+1,1] = energy(u,v,w)

    print(t,e_kin[n+1,1])
 
    # Energy dissipation
    dissp[n,0] = t - 0.5*dt
    dissp[n,1] = (e_kin[n+1,1]-e_kin[n,1])/dt
    
end_time = time.time()
runtime = end_time - start_time
print("Runtime [s] =", runtime)

k_dns, Ek_dns = energy_spectrum(u, v, w, lx, ly, lz)



grid_tag = f"grid2e{q}" #for naming files
def clean(s):
    return str(s).replace('.', '')
cfl_tag = f"cfl{clean(cfl)}"

folderDNS = f"results/DNS_{cfl_tag}/{grid_tag}"
os.makedirs(folderDNS, exist_ok=True)

np.savetxt(f"{folderDNS}/runtime_dns_{grid_tag}_{cfl_tag}.dat", np.array([runtime]))
np.savetxt(f"{folderDNS}/energy_dns_{grid_tag}_{cfl_tag}.dat",e_kin,fmt='%.12f')
np.savetxt(f"{folderDNS}/dissipation_dns_{grid_tag}_{cfl_tag}.dat",dissp,fmt='%.12f')
np.savetxt(f"{folderDNS}/spectrum_dns_{grid_tag}_{cfl_tag}.dat", np.column_stack((k_dns, Ek_dns)))