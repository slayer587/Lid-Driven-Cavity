import numpy as np
import matplotlib.pyplot as plt

# Parameters
Re = 1000
Nx = Ny = 129
Lx = Ly = 1
dx = dy = Lx / (Nx-1)
t = 0
sor = 1
iteration = 0
omega = np.zeros((Nx+2, Ny+2))
omega_new = np.zeros((Nx+2, Ny+2))
psi = np.zeros((Nx+2, Ny+2))
u = np.zeros((Nx+2, Ny+2))
v = np.zeros((Nx+2, Ny+2))
u[0,:] = 1
dt = 0.1 * min(dx/u.max(), Re*dx**2/4)
print("dx, dt", dx, dt)
print("Running...") 

# Data set from Ghia
if Re == 100:
    U = np.array([0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, 
              -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0000])
    y1 = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 
               0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
    V = np.array([0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, 
              -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0])
    x1 = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 
               0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])

if Re == 400:
    U = np.array([0,-0.18109,-0.20196,-0.2222,-0.29730,-0.38289,-0.27805,-0.10648,-0.0608,
                -0.05702,0.18719,0.33304,0.46604,0.51117,0.57492,0.65928,1.0000])
    y1 = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 
      0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
    V = np.array([0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, 
     -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21338, 0])
    x1 = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 
      0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])
    
if Re == 1000:
    U = np.array([0, -0.18109, -0.20196, -0.2222, -0.29730, -0.38289, -0.27805, -0.10648, 
     -0.0608, -0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.0000])
    y1 = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 
      0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
    V = np.array([0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, 
     -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21338, 0])
    x1 = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 
      0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])
    
if Re == 3200:
    U = np.array([0,-0.32407,-0.35344,-0.37827,-0.41933,-0.34323,-0.24427,-0.86636,-0.04272,
         0.07156,0.19791,0.34682,0.46101,0.46547,0.48296,0.53236,1])
    y1 = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 
         0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
    V = np.array([0,0.39560,0.40917,0.41906,0.42768,0.37119,0.29030,0.28188,0.00999,-0.31184,
        -0.37401,-0.44307,-0.54053,-0.52357,-0.47425,-0.39017,0])
    x1 = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 
      0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])

# Boundary Condition
omega[0,:] = - 8 / dy - 16 * psi[1,:] / dy**2 - omega[1,:]
psi[0,:] = dy + psi[1,:]

# Solver
while True:
    t += dt
    iteration += 1

    # Solving for psi
    psi_new = np.copy(psi)
    iteration_psi = 0
    convergence = 0
    while True:
        iteration_psi += 1
        psi_temp = np.copy(psi_new)
        for j in range(1, Nx+1):
            for i in range(1, Nx+1):
                psi_new[i,j] = (1 - sor) * psi_temp[i,j] + sor * (0.25 * (omega[i,j] * dx**2 + psi_new[i-1,j] + psi_new[i+1,j] + psi_new[i,j-1] + psi_new[i,j+1]))
        error = np.abs(np.max(psi_new - psi_temp))
        if iteration_psi > 2000:
            convergence = 1
            break
        if error < 1e-3:
            print("Converged after", iteration_psi, "iteration at t =", t, "with SOR", sor)
            break     
    if convergence == 1:
        print("Jacobi didn't converging after", iteration, "iterations")
        break
        
    # Updating Boundary condition for psi
    psi_new[:,0] = psi_new[:,1]
    psi_new[:,Nx+1] = psi_new[:,Nx]
    psi_new[Nx+1,:] = psi_new[Nx,:]
    psi_new[0,:] = dy + psi_new[1,:]
               
    # Solving for omega using Explicit Method
    for j in range(1, Nx+1):
        for i in range(1, Nx+1):
            # Convective terms
            convective_x = u[i,j] * (omega[i,j+1] - omega[i,j-1]) / (2 * dx)
            convective_y = v[i,j] * (omega[i-1,j] - omega[i+1,j]) / (2 * dy)
            # Diffusive terms
            diffusive_x = (omega[i,j+1] - 2 * omega[i,j] + omega[i,j-1]) / dx**2
            diffusive_y = (omega[i-1,j] - 2 * omega[i,j] + omega[i+1,j]) / dy**2
            # Update omega using both terms
            omega_new[i,j] = omega[i,j] - dt * (convective_x + convective_y) + (dt / Re) * (diffusive_x + diffusive_y)
        
    # Update Boundary Condition for omega
    omega_new[:,0] = - 16 * psi_new[:,0] / dx**2 - omega_new[:,1]
    omega_new[:,Nx+1] = - 16 * psi_new[:,Nx] / dx**2 - omega_new[:,Nx]
    omega_new[Nx+1,:] = - 16 * psi_new[Nx+1,:] / dy**2 - omega_new[Nx,:]
    omega_new[0,:] = - 8 / dy - 16 * psi_new[1,:] / dy**2 - omega_new[1,:]
    
       
    # Update velocities
    for j in range(1, Nx+1):
        for i in range(1, Nx+1):
            u[i,j] = (psi_new[i-1,j] - psi_new[i+1,j]) / (2 * dy)
            v[i,j] = -(psi_new[i,j+1] - psi_new[i,j-1]) / (2 * dx)

    convergence = np.abs(np.max(omega_new - omega))
    omega = np.copy(omega_new)
    psi = np.copy(psi_new)
    print("Time", t,"\nJacobi Converged after", iteration_psi, "iterations\nIteration", iteration, "and Convergence error", convergence) 
    # Plotting at every 1sec   
    if t % 1 < dt:
        print("Time", t,"\nJacobi Converged after", iteration_psi, "iterations\nIteration", iteration, "and Convergence error", convergence)  
        x = np.linspace(0, Lx, Nx+2)
        y = np.linspace(0, Ly, Ny+2)
        X, Y = np.meshgrid(x, y,)
        
        # For Streamlines
        plt.figure(figsize=(6, 6))
        contour_plot = plt.contour(X, Y, psi, levels=30, colors='black', linestyles='solid', linewidths=0.5)
        plt.title(f'Re = {Re} at timestep {t}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().invert_yaxis()
        plt.show()

        # For Comparing U-velocity with Ghia
        plt.figure(figsize=(6, 6))
        plt.plot(y1, U, 'o', color='red', label='Ghia results', linewidth=2)
        plt.plot(y, u[::-1, (Ny + 1) // 2], '-g', label='Present code results', linewidth=2)
        plt.xlabel('Y')
        plt.ylabel('U velocity')
        plt.title(f'Validation with Ghia for Re = {Re} at timestep {t}')
        plt.legend()
        plt.grid(True)
        plt.show()
            
        # For Comparing V-velocity with Ghia
        plt.figure(figsize=(6, 6))
        plt.plot(x1, V, 'o', color='red', label='Ghia results', linewidth=2)
        plt.plot(x, v[(Nx + 1) // 2, :], '-g', label='Current code results', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('V velocity')
        plt.title(f'Validation with Ghia for Re = {Re} at timestep {t}')
        plt.legend()
        plt.grid(True)
        plt.show()
    if iteration > 5 and convergence < 1e-6 or t >120:
        print("Final Convergence in", iteration, "iteration at t =", t)
        break 
