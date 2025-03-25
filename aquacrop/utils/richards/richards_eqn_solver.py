import numpy as np
from scipy.sparse.linalg import spsolve

def thetaf(psi,pars):
    '''Calculate volumetric water content using the van Genuchten model'''
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Se[psi>0.]=1.0
    return pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se

# def C(psi,pars):
#     '''Calculate specific moisture capacity (C)'''
#     Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
#     Se[psi>0.]=1.0
#     dSedh=pars['alpha']*pars['m']/(1-pars['m'])*Se**(1/pars['m'])*(1-Se**(1/pars['m']))**pars['m']
#     return Se*pars['Ss']+(pars['thetaS']-pars['thetaR'])*dSedh

def C(psi, pars):
    "Calculate specific moisture capacity C(h) = d(theta)/dh"
    nume = pars['m'] * pars['n'] * pars['alpha'] * (pars['alpha'] * np.abs(psi))**(pars['n']-1)
    denom = (1 + (pars['alpha'] * np.abs(psi))**(pars['n']))**(pars['m']+1)
    return -(pars['thetaS'] - pars['thetaR']) * nume/denom * psi/(np.abs(psi)+1e-10)

def K(psi,pars):
    '''Compute hydraulic conductivity via the Mualem model'''
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Se[psi>0.]=1.0
    return pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2

def simulate_rainfall(step):
  """Simulates rainfall for a given hour.

  Args:
    step: The current hour (0-23).

  Returns:
    Rainfall in m^3 per m^2 per hour.  A simple example with no rain for most of the day,
    and a brief period of rain in the late afternoon/early evening.
  """
  if step > 100:
      return 0.0
  if step < 70 and 17 <= step <= 19:  # Rain between 5 PM and 7 PM
    return 0.010/3600  # Example: 2mm/hour rainfall = 0.002 m/hr
  else:
    return 0.0

def simulate_irrigation(step):
  """Simulates irrigation for a given hour.

  Args:
    step: The current hour (0-23).

  Returns:
    Irrigation in m^3 per m^2 per hour.  A simple example with irrigation in the
    early morning and late evening.
  """
  if step > 100:
      return 0.0
  if step < 60 and 6 <= step <= 7:    # Irrigate between 6 AM and 7 AM
    return 0.002/3600  # Example: 2mm/hour irrigation
  elif step < 70 and 20 <= step <= 21: # Irrigate between 8 PM and 9 PM
    return 0.001/3600  # Example: 1mm/hour irrigation
  else:
    return 0.0


max_iter = 1000  # maximum possible picard iterations


# mean is used to calculate the conductivity on the surface
def mean(K_i, K_j):
    return (K_i + K_j)/2



# maximum infiltration possible at top layer
def compute_q_max(K_half_top, h0, h1, dz):
    return K_half_top * ((h1 - h0) / dz + 1)


# iteratively calculate boundary value using Newton's Raphson method
def calculate_h0(rainfall, irrigation, h_curr, pars, dz):
    R = rainfall + irrigation
    if R > pars['Ks']:
        return 0.0
    runoff = max(R - pars['Ks'], 0.0)
    h0 = h_curr[0]
    h1 = h_curr[1]
    omega = 0.5
    for i in range(1000):
        Kh0 = K(h0,pars)
        h_new = h1 + dz * (- R / Kh0 - 1)
        delta = h_new - h0
        h0 += omega * delta
        if abs(delta) < 1e-10:
            return h0
    raise ValueError("Fixed point iteration did not converge")

# Apply boundary condition directly into the system of linear equations.
def apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter):
    """
    Update surface matric potential (h0) and calculate runoff during a Picard iteration.

    Args:
        h_current (array): Current matric potential values [m] (including h0 at index 0).
        K_half (array): Hydraulic conductivity at cell interfaces [m/hr] (K_half[0] = surface conductivity).
        R: rainfall (float): Rainfall rate [m/hr] + irrigation (float): Irrigation rate [m/hr].
        dz (float): Vertical grid spacing [m].
        pars (dict): Soil parameters (must include 'Ks' for saturated conductivity).

    Returns:
        tuple: (h0_updated, runoff)
            h0_updated (float): Updated surface matric potential [m].
            runoff (float): Runoff rate [m/hr].
    """
    # Sink = irrigation + precipitation - evaporation - deep_percolation
    # Compute maximum infiltration capacity
    Ks = pars['Ks']

    # Extract current h0 and h1 (first subsurface node)

    infil_cap = -K_current[0] * ((h_current[1]- h_current[0])/dz + 1)
    q_unsat = Ks
    if R <= q_unsat:
        flux = R
    else:
        flux = q_unsat

    sink = flux/dz
    # Update b with flux
    b[0] += sink

    # Keep h0 at or below zero
    runoff = max(0, R - flux)

    return runoff


def compute_deep_percolation(K_half, h_bottom, h_current, dz, dt):
    """
    Compute deep percolation flux [m/hr] and volume [m] for a time step.

    Args:
        K_half_bottom: Hydraulic conductivity at the bottom interface (i=N-1/2).
        h_bottom: Pressure head at the bottom boundary (Dirichlet value).
        h_prev_last: Pressure head in the last soil compartment (i=N-1).
        dz: Spatial discretization depth [m].
        dt: Time step [hr].
    Returns:
        q_bottom: Deep percolation flux [m/hr].
        percolation_volume: Cumulative percolation over Δt [m].
    """
    # Compute Darcy flux at bottom boundary (positive downward)
    q_bottom = K_half[-1] * 1.0

    # Convert flux to volume per unit area over Δt (e.g., meters of water)
    percolation_volume = q_bottom * dt

    return q_bottom, percolation_volume


def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt):
    '''
    K_half : Value of K at boundary of each soil compartments
    C_val: value of C at each compartment
    theta_val: value of theta at each compartment
    h_current: value of h at each compartment
    dz = depth of one compartment
    dt = timestep

    Output:
    A is the tri-diagonal matrix where each row corresponds to each compartment. The top compartment has no K_{i-1/2}, since it has only 2 non zero values
    and bottom compartment doesnt have K_{i+1/2} so it also has only 2 non zero values.
    '''
    n = len(h_curr) - 1
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # top boundary
    A[0, 0] = C_val[0] / dt + (K_half[0]) / dz ** 2
    A[0, 1] = - K_half[0] / dz ** 2
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt + K_half[0] / dz
    # middle compartments


    for i in range(1, n):
        A[i, i - 1] = - K_half[i - 1] / dz ** 2
        A[i, i] = C_val[i] / dt + (K_half[i] + K_half[i - 1]) / dz ** 2
        A[i, i + 1] = - K_half[i] / dz ** 2
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i] - K_half[i - 1]) / dz

    # bottom boundary
    A[n, n - 1] = - K_half[n - 1] / dz ** 2
    A[n, n] = C_val[n] / dt + (K_half[n - 1]) / dz ** 2

    # Let bottom boundary be 0 for now
    # b[n] = 0
    # free drainage
    b[n] = (theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + K_half[n - 1] / dz

    return A, b


def main():
    # Define some initial soil parameters
    # SandyLoam # source: https://www.gsshawiki.com/Infiltration:Parameter_Estimates
    # source: https://www.nature.com/articles/s41597-022-01481-5/tables/6
    # source: https://www.pc-progress.com/en/OnlineHelp/HYDRUS3/Hydrus.html?WaterFlowParameters.html
    pars = {}
    pars['thetaR'] = 0.131
    pars['thetaS'] = 0.387
    pars['alpha'] = 0.423
    pars['n'] = 2.06
    pars['m'] = 1 - 1 / pars['n']
    pars['Ks'] = 0.0496 / 3600 # m/s
    pars['neta'] = 0.5 # fixed value
    pars['Ss'] = 0.000001

    dz = 0.1
    Nz = 10
    #T = 100
    dt = 3600
    L = 1
    z = np.linspace(0, L, Nz)
    #time = np.arange(0, T + dt, dt)
    Nt = 150

    # Array to store the pressure head at each node at each time step
    psi = np.zeros((Nz, Nt))
    psi[:, 0] = -z

    # Array to store the water content in each node at each time step
    theta_val = np.zeros((Nz, Nt))
    runoff_val = np.zeros(Nt)
    deep_percolation_val = np.zeros(Nt)

    # initialize the initial water content in each compartment by executing theta function.
    K_val = np.zeros((Nz, Nt))
    C_val = np.zeros((Nz, Nt))
    tolerance = 1e-10

    for step in range(Nt):
        # returns (m^3 per sq.m per hour)
        # takes current hour as input
        print(f'time = {step} hour')
        rainfall = simulate_rainfall(step % 24)
        irrigation = simulate_irrigation(step % 24)
        R = rainfall + irrigation
        if step > 0:
            theta_prev = theta_val[:, step - 1]
            h_current = psi[:, step - 1].copy()

        else:
            h_current = np.full(Nz, -67.5)
            h_current[0] = -20.0
            theta_prev = thetaf(h_current + 5.0, pars)


        theta_current = thetaf(h_current, pars)
        K_current = K(h_current, pars)
        C_current = C(h_current, pars)

        # at each timestep, update the value of theta, K and C for each compartment
        for iter in range(max_iter):
            # at each step of picard iteration
            # values at iter = m
            theta_val[:, step] = thetaf(h_current, pars)
            K_val[:, step] = K(h_current, pars)
            C_val[:, step] = C(h_current, pars)

            K_step = K_val[:, step]
            # first layer is bit complicated as we need to set up boundary condition.
            K_half = mean(K_step[:-1], K_step[1:])
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, dz, dt)


            if R > 0:
                runoff = apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter)
            else:
                runoff = 0.0

            h_new = spsolve(A, b)
            h_new[h_new > 0] = 0.0
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance
            max_diff = np.max(np.abs(h_new - h_current))
            if max_diff < tolerance:
                break

            # relaxation factor
            h_current = h_new * 0.6 + h_current * 0.4

            theta_current = thetaf(h_current, pars)
            K_current = K(h_current, pars)
            C_current = C(h_current, pars)

        # update theta/K/C
        theta_val[:, step] = theta_current
        K_val[:, step] = K_current
        C_val[:, step] = C_current
        runoff_val[step] = runoff # unit: m/hr
        psi[:, step] = h_current

        # calculate deep percolation
        h_bottom = h_current[-1]
        q_bottom, deep_percolation = compute_deep_percolation(K_half, h_bottom, h_current, dz, dt)
        deep_percolation_val[step] = deep_percolation

    print('completed simulation')


if __name__ == '__main__':
    main()