#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Question 1

#####################################################################
# SOR (successive over relaxation) for the 2D Poisson equation       #
# lap(phi) = -rho is solved with a red-black ordering, so every      #
# half sweep can be written with numpy arrays.                       #
# BC = 'dirichlet' -> phi = 0 on the edges of the grid               #
# BC = 'periodic'  -> the grid wraps around itself (np.roll)          #
#####################################################################

def SOR_poisson(rho, dx, omega, BC = 'dirichlet', tol = 1e-8, max_it = 50000):

    #Initilizing varibles / Arrays
    N = rho.shape[0]
    phi = np.zeros((N, N))

    i_mat, j_mat = np.meshgrid(np.arange(N), np.arange(N), indexing = 'ij')

    if BC == 'dirichlet':
        #Only the inner points are updated, the edges stay phi = 0
        update = np.zeros((N, N), dtype = bool)
        update[1:N-1, 1:N-1] = True
    else:
        #For periodic BC every point is updated
        update = np.ones((N, N), dtype = bool)

    red = update & ((i_mat + j_mat) % 2 == 0)
    black = update & ((i_mat + j_mat) % 2 == 1)

    #Relaxation loop

    for it in range(max_it):

        max_change = 0

        for mask in [red, black]:

            #Sum of the 4 neighbours (np.roll gives the periodic wrap,
            #for the dirichlet case the edges are never updated so they stay 0)
            neighbours = (np.roll(phi, 1, axis = 0) + np.roll(phi, -1, axis = 0)
                          + np.roll(phi, 1, axis = 1) + np.roll(phi, -1, axis = 1))

            phi_GS = (1/4)*(neighbours + dx**2*rho)
            change = omega*(phi_GS - phi)

            phi[mask] = phi[mask] + change[mask]
            max_change = max(max_change, np.max(np.abs(change[mask])))

        if max_change < tol:
            break

    if BC == 'periodic':
        #The potential is only defined up to a constant, we choose <phi> = 0
        phi = phi - np.mean(phi)

    return(phi, it + 1)


def free_space_potential(x, y, x_q, y_q, q, r_ref):
    #Potential of a line charge in 2D: phi = -(q/2pi) ln(r) + C,
    #the constant is chosen such that phi(r_ref) = 0
    r = np.sqrt((x - x_q)**2 + (y - y_q)**2)
    r[r == 0] = np.nan
    return((q/(2*np.pi))*np.log(r_ref/r))


#############################
# a) Dirichlet BC           #
#############################

L = 1.0
N = 101                     #grid points including the edges
dx = L/(N-1)
q = 1.0

x_grid = np.linspace(0, L, N)
X, Y = np.meshgrid(x_grid, x_grid, indexing = 'ij')

rho = np.zeros((N, N))
i_c = N//2
rho[i_c, i_c] = q/dx**2     #single charge at the center cell

omega = 2/(1 + np.sin(np.pi/(N-1)))     #optimal omega for a square grid

phi_D, it_D = SOR_poisson(rho, dx, omega, BC = 'dirichlet')
print(f'Dirichlet BC: converged after {it_D} iterations')

phi_free = free_space_potential(X, Y, x_grid[i_c], x_grid[i_c], q, r_ref = L/2)

levels = np.linspace(-0.1, 0.4, 26)     #the colors are saturated near the charge

plt.figure(figsize = (11, 4.5))

plt.subplot(1,2,1)
plt.contourf(X, Y, phi_D, levels = levels, cmap = 'viridis', extend = 'both')
plt.colorbar(label = r'$\Phi$')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SOR, Dirichlet BC ($\\Phi$ = 0 on the edges)')

plt.subplot(1,2,2)
plt.contourf(X, Y, phi_free, levels = levels, cmap = 'viridis', extend = 'both')
plt.colorbar(label = r'$\Phi$')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Free space line charge, $\Phi = \frac{q}{2\pi}\ln(\frac{0.5}{r})$')

plt.tight_layout()
plt.show()

#Cut along y = 0.5 and along the diagonal

r_cut = x_grid[i_c:] - x_grid[i_c]
r_diag = np.sqrt(2)*r_cut

plt.plot(r_cut[1:], phi_D[i_c+1:, i_c], 'o', markersize = 3, label = r'SOR Dirichlet, along $y = 0.5$')
plt.plot(r_diag[1:], np.diag(phi_D)[i_c+1:], 's', markersize = 3, label = 'SOR Dirichlet, along the diagonal')
r_plot = np.linspace(dx, np.sqrt(2)/2, 300)
plt.plot(r_plot, (q/(2*np.pi))*np.log(0.5/r_plot), 'k', label = 'Free space line charge')
plt.axhline(0, color = 'gray', lw = 0.5)
plt.xlabel('distance from the charge r')
plt.ylabel(r'$\Phi$')
plt.title('Dirichlet BC vs. free space')
plt.legend()
plt.show()


#############################
# b) Periodic BC            #
#############################

N_p = 100                   #with periodic BC the point x = 1 is the point x = 0
x_grid_p = np.arange(N_p)*dx
X_p, Y_p = np.meshgrid(x_grid_p, x_grid_p, indexing = 'ij')
i_c_p = N_p//2

rho_p = np.zeros((N_p, N_p))
rho_p[i_c_p, i_c_p] = q/dx**2

#A periodic Poisson problem has a solution only if the total charge is 0,
#so a uniform neutralizing background is subtracted
rho_p = rho_p - np.mean(rho_p)

phi_P, it_P = SOR_poisson(rho_p, dx, omega, BC = 'periodic')
print(f'Periodic BC: converged after {it_P} iterations')

plt.figure(figsize = (11, 4.5))

plt.subplot(1,2,1)
plt.contourf(X, Y, phi_D, levels = levels, cmap = 'viridis', extend = 'both')
plt.colorbar(label = r'$\Phi$')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dirichlet BC')

plt.subplot(1,2,2)
plt.contourf(X_p, Y_p, phi_P, levels = levels, cmap = 'viridis', extend = 'both')
plt.colorbar(label = r'$\Phi$')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Periodic BC (with neutralizing background)')

plt.tight_layout()
plt.show()

#Comparison along y = 0.5, every curve is shifted to be 0 at r_ref = 0.1
#(the constant of the potential is arbitrary)

i_ref = int(round(0.1/dx))

plt.plot(x_grid, phi_D[:, i_c] - phi_D[i_c + i_ref, i_c], label = 'Dirichlet BC')
plt.plot(x_grid_p, phi_P[:, i_c_p] - phi_P[i_c_p + i_ref, i_c_p], label = 'Periodic BC')
phi_free_cut = free_space_potential(x_grid, 0.5*np.ones(N), 0.5, 0.5, q, r_ref = 0.1)
plt.plot(x_grid, phi_free_cut, 'k--', label = 'Free space')
plt.xlabel('x  (y = 0.5)')
plt.ylabel(r'$\Phi - \Phi(r = 0.1)$')
plt.title('Dirichlet vs. periodic BC')
plt.legend()
plt.show()


#%% Question 2

####################################################################
# First order upwind schemes for Burgers' equation (u > 0 here)     #
# conservative = False ->  u_t + u u_x = 0                          #
# conservative = True  ->  u_t + (u^2/2)_x = 0                      #
# Outflow BC: the ghost cells copy the value of the edge cells.     #
####################################################################

def burgers_upwind(u_i, x_min, x_max, t_f, CFL, conservative, t_out = [], u_shock = 1.5, g_points = 1):

    #Initilizing varibles / Arrays
    N = len(u_i) + 2*g_points
    dx = (x_max - x_min)/len(u_i)
    x_cells = x_min + (np.arange(len(u_i)) + 0.5)*dx

    u = np.zeros(N)
    u[1:N-1] = u_i
    u[0] = u[1]
    u[N-1] = u[N-2]

    t = 0
    t_vec = [0]
    x_shock_vec = [x_cells[np.argmax(u_i < u_shock)]]
    snapshots = []
    t_out = list(t_out)

    #Time steps loop

    while t < t_f:

        dt = CFL*dx/np.max(np.abs(u))

        #landing exactly on the requested output times
        if len(t_out) > 0 and t + dt > t_out[0]:
            dt = t_out[0] - t
        if t + dt > t_f:
            dt = t_f - t

        if conservative:
            F = 0.5*u**2
            u[1:N-1] = u[1:N-1] - (dt/dx)*(F[1:N-1] - F[0:N-2])
        else:
            u[1:N-1] = u[1:N-1] - (dt/dx)*u[1:N-1]*(u[1:N-1] - u[0:N-2])

        u[0] = u[1]
        u[N-1] = u[N-2]

        t = t + dt
        t_vec.append(t)

        #shock position = first zone where u drops below u_shock
        x_shock_vec.append(x_cells[np.argmax(u[1:N-1] < u_shock)])

        if len(t_out) > 0 and np.isclose(t, t_out[0]):
            snapshots.append(u[1:N-1].copy())
            t_out.pop(0)

    return(x_cells, snapshots, np.array(t_vec), np.array(x_shock_vec))


def initial_conditions(n, x_min, x_max):
    dx = (x_max - x_min)/n
    x_cells = x_min + (np.arange(n) + 0.5)*dx
    u = np.ones(n)
    u[x_cells < 0.5] = 2.0
    return(u)


x_min = 0
x_max = 1
C = 0.5
t_f = 0.25
t_out = [0.05, 0.10, 0.15, 0.20, 0.25]
colors = ['blue', 'purple', 'red', 'orange', 'green']

N = 256
u_0 = initial_conditions(N, x_min, x_max)

titles = ["a) Upwind, non conservative form $u_t + u u_x = 0$",
          "b) Upwind, conservative form $u_t + (u^2/2)_x = 0$"]

for k, conservative in enumerate([False, True]):

    x_cells, snapshots, t_vec, x_shock_vec = burgers_upwind(u_0, x_min, x_max, t_f, C, conservative, t_out)

    plt.plot(x_cells, u_0, 'k--', label = 't = 0')
    for i in range(len(t_out)):
        plt.plot(x_cells, snapshots[i], color = colors[i], label = f't = {t_out[i]:.2f}')
        plt.axvline(0.5 + 1.5*t_out[i], color = colors[i], ls = ':', lw = 0.8)

    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(titles[k] + f', N = {N}')
    plt.legend()
    plt.show()


#############################
# c) Shock speed            #
#############################

N_vec = [128, 256, 512, 1024]
s_exact = (2.0 + 1.0)/2      #Rankine-Hugoniot: s = (u_L + u_R)/2

method_names = ['Non conservative', 'Conservative']
markers = ['o', 's']

for k, conservative in enumerate([False, True]):

    plt.subplot(1,2,k+1)
    print(method_names[k])

    for n in N_vec:

        u_0 = initial_conditions(n, x_min, x_max)
        x_cells, snapshots, t_vec, x_shock_vec = burgers_upwind(u_0, x_min, x_max, t_f, C, conservative)

        #fitting a straight line after the initial transient
        fit_range = t_vec > 0.05
        s_num = np.polyfit(t_vec[fit_range], x_shock_vec[fit_range], 1)[0]
        print(f'   N = {n}: shock speed = {s_num:.4f}')

        plt.plot(t_vec, x_shock_vec, label = f'N = {n}, s = {s_num:.3f}')

    plt.plot(t_vec, 0.5 + s_exact*t_vec, 'k--', label = f'Exact, s = {s_exact}')
    plt.xlabel('t')
    plt.ylabel('shock position')
    plt.title(method_names[k])
    plt.legend()

plt.tight_layout()
plt.show()


#%% Question 3

###########################################################
# Boris push: advancing v by dt for q/m = 1 in 2D, where   #
# E = (E_x, E_y) and B = B_z z^                            #
###########################################################

def boris_push(v, E, B_z, dt):

    v_minus = v + 0.5*dt*E                      #half electric kick

    t_B = 0.5*dt*B_z                            #magnetic rotation
    s_B = 2*t_B/(1 + t_B**2)
    v_prime = v_minus + np.array([v_minus[1]*t_B, -v_minus[0]*t_B])      #v_minus x t
    v_plus = v_minus + np.array([v_prime[1]*s_B, -v_prime[0]*s_B])       #v_prime x s

    return(v_plus + 0.5*dt*E)                   #second half electric kick


#####################################################################
# Leapfrog: x is known at integer steps, v at half integer steps    #
#   v_(n+1/2) = Boris(v_(n-1/2))                                    #
#   x_(n+1)   = x_n + dt*v_(n+1/2)                                  #
#####################################################################

def leapfrog_boris(x_0, v_0, E_field, B_field, dt, it_num):

    #Initilizing varibles / Arrays
    x_vec = np.zeros((it_num + 1, 2))
    v_vec = np.zeros((it_num + 1, 2))      #v_vec[n] = v at t = (n - 1/2)dt
    t_vec = np.arange(it_num + 1)*dt

    x_vec[0] = x_0

    #v at t = -dt/2 is found by pushing v_0 half a step backwards
    v_vec[0] = boris_push(np.array(v_0, dtype = float), E_field(x_vec[0], 0), B_field(x_vec[0], 0), -dt/2)

    #Time steps loop

    for n in range(it_num):
        v_vec[n+1] = boris_push(v_vec[n], E_field(x_vec[n], t_vec[n]), B_field(x_vec[n], t_vec[n]), dt)
        x_vec[n+1] = x_vec[n] + dt*v_vec[n+1]

    return(t_vec, x_vec, v_vec)


def gyration_frequency(t_vec, x_vec):
    #Times in which x - <x> crosses zero upwards, interpolated linearly
    x_rel = x_vec[:, 0] - np.mean(x_vec[:, 0])
    idx = np.where((x_rel[:-1] < 0) & (x_rel[1:] >= 0))[0]
    t_cross = t_vec[idx] - x_rel[idx]*(t_vec[idx+1] - t_vec[idx])/(x_rel[idx+1] - x_rel[idx])
    T = np.mean(np.diff(t_cross))
    return(2*np.pi/T, T)


x_0 = np.array([0.0, 1.0])
v_0 = np.array([1.0, 0.0])
B_0 = 1.0

T_L = 2*np.pi/B_0           #theoretical Larmor period (q/m = 1)
dt = T_L/100
periods = 10
it_num = periods*100


###########################################
# a) Uniform B = 1 z^, no electric field  #
###########################################

E_field = lambda r, t: np.array([0.0, 0.0])
B_field = lambda r, t: B_0

t_vec, x_vec, v_vec = leapfrog_boris(x_0, v_0, E_field, B_field, dt, it_num)

center = np.mean(x_vec[:-1], axis = 0)                  #average over full periods
r_num = np.mean(np.sqrt(np.sum((x_vec - center)**2, axis = 1)))
omega_num, T_num = gyration_frequency(t_vec, x_vec)

v_perp = np.sqrt(np.sum(v_0**2))
r_L = v_perp/B_0                                        #r_L = m v / (q B)
omega_L = B_0                                           #omega_L = q B / m
omega_boris = (2/dt)*np.arctan(omega_L*dt/2)            #exact frequency of the Boris rotation

print(f'center of the orbit = ({center[0]:.2e}, {center[1]:.2e})')
print(f'radius:    numerical = {r_num:.6f}, Larmor = {r_L:.6f}')
print(f'frequency: numerical = {omega_num:.6f}, Larmor = {omega_L:.6f}, (2/dt)arctan(w dt/2) = {omega_boris:.6f}')

plt.plot(x_vec[:, 0], x_vec[:, 1], label = 'Boris leapfrog')
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(r_L*np.cos(theta), r_L*np.sin(theta), 'k--', lw = 1, label = f'Larmor circle, $r_L$ = {r_L}')
plt.plot(x_0[0], x_0[1], 'ro', label = 'Starting point')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Uniform $B = \\hat z$, E = 0\n$r$ = {r_num:.5f}, $\\omega$ = {omega_num:.5f}')
plt.legend()
plt.show()


##################################################
# b) Uniform B = 1 z^ and E = -0.5 y^            #
##################################################

E_0 = np.array([0.0, -0.5])
E_field = lambda r, t: E_0

periods = 20
it_num = periods*100
t_vec, x_vec, v_vec = leapfrog_boris(x_0, v_0, E_field, B_field, dt, it_num)

#Drift velocity: the guiding center R = r + v x B / B^2 obeys dR/dt = E x B / B^2
#exactly (the gyration cancels out), so the slope of R(t) gives the drift velocity.
#v at integer steps is the average of the two neighbouring half steps.
v_int = 0.5*(v_vec[:-1] + v_vec[1:])
R_gc = x_vec[:-1] + np.column_stack([v_int[:, 1]*B_0, -v_int[:, 0]*B_0])/B_0**2
v_d_num = np.array([np.polyfit(t_vec[:-1], R_gc[:, 0], 1)[0], np.polyfit(t_vec[:-1], R_gc[:, 1], 1)[0]])

#Theory: v_E = E x B / B^2
v_d_theory = np.array([E_0[1]*B_0, -E_0[0]*B_0])/B_0**2

print(f'drift velocity: numerical = ({v_d_num[0]:.5f}, {v_d_num[1]:.5f}), E x B / B^2 = ({v_d_theory[0]}, {v_d_theory[1]})')

n_plot = 6*100 + 1          #plotting only the first 6 periods

plt.figure(figsize = (10, 4))
plt.plot(x_vec[:n_plot, 0], x_vec[:n_plot, 1], label = 'Boris leapfrog')
plt.plot(R_gc[:n_plot, 0], R_gc[:n_plot, 1], 'k--', lw = 1, label = r'Guiding center $\vec R = \vec r + \vec v \times \vec B / B^2$')
plt.plot(x_0[0], x_0[1], 'ro', label = 'Starting point')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Uniform $B = \\hat z$, $E = -0.5\\hat y$: numerical drift = ({v_d_num[0]:.4f}, {v_d_num[1]:.4f})')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.25), ncol = 3)
plt.show()

# %%
