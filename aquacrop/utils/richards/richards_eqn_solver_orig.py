import math

import numpy as np
from typing import Tuple, TYPE_CHECKING

from scipy.sparse.linalg import spsolve
if TYPE_CHECKING:
    # Important: classes are only imported when types are checked, not in production.
    from numpy import ndarray
    from aquacrop.entities.clockStruct import ClockStruct
    from aquacrop.entities.initParamVariables import InitialCondition
    from aquacrop.entities.paramStruct import ParamStruct
    from aquacrop.entities.output import Output
    from aquacrop.entities.crop import Crop

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

# def K(psi,pars):
#     '''Compute hydraulic conductivity via the Mualem model'''
#     Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
#     Se[psi>0.]=1.0
#     return pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2

def K(psi,pars):
    alpha = pars['alpha']
    n = pars['n']
    m = pars['m']
    Se = (1 + (-psi*alpha)**n)**(-m)
    nume = (1 - (alpha * (-psi))**(n-1)* Se )**2
    denom = (1 + (alpha * (-psi))**n )**(m/2)
    val = (nume/denom) * pars['Ks']
    return val.values

def simulate_rainfall(step):
  """Simulates rainfall for a given hour.

  Args:
    step: The current hour (0-23).

  Returns:
    Rainfall in m^3 per m^2 per hour.  A simple example with no rain for most of the day,
    and a brief period of rain in the late afternoon/early evening.
  """
  if (step % 18*24) in [0,1,2]:  # Rain between 5 PM and 7 PM
    return 0.005  # Example: 2mm/hour rainfall = 0.005 m/hr
  if (step % 10*24) in [0,1,2]:  # Rain between 5 PM and 7 PM
    return 0.002  # Example: 2mm/hour rainfall = 0.002 m/hr
  else:
    return -0.1/1000 # 0.1 mm/hr (average evaporation over growing months)

def simulate_irrigation(step):
  """Simulates irrigation for a given hour.

  Args:
    step: The current hour (0-23).

  Returns:
    Irrigation in m^3 per m^2 per hour.  A simple example with irrigation in the
    early morning and late evening.
  """
  if step > 1500:
      return 0.0
  if step % (7*24) in [0,1,2]:    # Irrigate 6mm every week.
    return 0.002  # Example: 2mm/hour irrigation
  else:
    return 0.0


max_iter = 1000  # maximum possible picard iterations


# mean is used to calculate the conductivity on the surface
def mean(K_i, K_j):
    return (K_i + K_j)/2

def calculate_top_flux_runoff(R, h_current, K_current, dz, pars):
    """
    Calculates the actual surface flux (Darcy convention, upward positive)
    and runoff based on the potential flux rate R and the current soil state.
    Updates h_current[0] in-place if ponding occurs.

    Args:
        R (float): Potential flux rate (+ve potential infiltration, -ve potential evaporation).
                   NOTE: This R follows meteorological convention!
        h_current (np.array): Current pressure heads (h[0] is surface).
        K_current (np.array): Current hydraulic conductivities corresponding to h_current.
        dz (float): Spacing between surface node (0) and first node below (1).
                    Must be positive.
        pars (dict): Dictionary containing parameters, requires 'Ks'.

    Returns:
        tuple: (q_darcy, runoff)
            q_darcy (float): The actual Darcy flux across the surface boundary
                             (positive upward, negative downward).
            runoff (float): The runoff generated (always >= 0).
    """
    # --- Input Validation and Initialization ---
    if dz <= 0:
        raise ValueError("dz must be positive to calculate gradient.")
    if len(h_current) < 2 or len(K_current) < 1:
         raise ValueError("h_current must have at least 2 elements, K_current at least 1.")

    # Only execute meaningful calculations if R is non-zero
    if np.isclose(R, 0.0):
        return 0.0, 0.0  # No potential flux -> no actual flux, no runoff

    Ks = pars['Ks']
    h0 = h_current[0]
    h1 = h_current[1] # Head at first node below surface
    K0 = K_current[0] # Conductivity corresponding to h0

    runoff = 0.0
    q_darcy = 0.0 # Initialize actual Darcy flux (downward positive)

    # Calculate potential flux the soil can transmit based on current gradient
    # grad_H = dh/dz + 1 = (h1 - h0)/dz + 1
    # q_soil_potential is the Darcy flux (upward positive) the soil could
    # sustain given the current h0, h1, K0 state.
    grad_H = (h1 - h0) / dz + 1
    q_soil_potential = -K0 * grad_H

    # --- Apply Boundary Condition Logic ---

    if R > 0: # Potential Infiltration (Downward Flow, q_darcy should be negative)
        potential_infiltration_rate = R # Magnitude of infiltration demand

        if h0 >= 0: # Case 1: Surface is already saturated/ponded
            # Flux limited by min of demand (R) and saturated conductivity (Ks)
            # Actual infiltration rate (magnitude)
            actual_inf_rate = min(potential_infiltration_rate, Ks)
            q_darcy = actual_inf_rate # Upward flux is positive
            runoff = potential_infiltration_rate - actual_inf_rate
            h_current[0] = 0.0 # Enforce h=0 at surface if saturated

        else: # Case 2: Surface is unsaturated (h0 < 0)
            # Calculate soil's infiltration capacity magnitude
            # Downward flow corresponds to negative q_soil_potential
            infiltration_capacity_magnitude = max(0.0, q_soil_potential)

            if potential_infiltration_rate <= infiltration_capacity_magnitude:
                # Case 2a: Soil can infiltrate all potential rain
                actual_inf_rate = potential_infiltration_rate
                q_darcy = actual_inf_rate
                runoff = 0.0
                # h_current[0] remains unchanged (still < 0)
            else:
                # Case 2b: Rain exceeds infiltration capacity -> ponding occurs
                actual_inf_rate = infiltration_capacity_magnitude
                q_darcy = actual_inf_rate
                runoff = potential_infiltration_rate - actual_inf_rate
                h_current[0] = 0.0 # Surface becomes saturated

    elif R < 0: # Potential Evaporation (Upward Flow, q_darcy should be positive)
        potential_evaporation_rate = - R # Magnitude of evaporation demand (positive)
        runoff = 0.0 # No runoff during evaporation

        if h0 >= 0: # Case 3: Surface is saturated/ponded
            # If saturated, assume soil can meet potential evaporation demand
            # (This might be energy-limited in reality, but often modeled this way)
            # Actual evaporation rate (magnitude)
            actual_evap_rate = potential_evaporation_rate
            q_darcy = - actual_evap_rate # Upward flux is negative flux
            # Note: If ponded depth > 0, evaporation removes water from pond first.
            # Setting h_current[0]=0 assumes pond is depleted or wasn't tracked.

        else: # Case 4: Surface is unsaturated (h0 < 0)
            # Calculate soil's exfiltration capacity magnitude (max upward flux)
            # Upward flow corresponds to positive q_soil_potential
            exfiltration_capacity_magnitude = q_soil_potential

            # Actual evaporation rate is limited by minimum of demand and capacity
            actual_evap_rate = min(potential_evaporation_rate, exfiltration_capacity_magnitude)
            q_darcy = - actual_evap_rate # Upward flux is negative flux
            # h_current[0] remains unchanged (still < 0)

    # Return the calculated actual Darcy flux and any generated runoff
    return q_darcy, runoff

# Apply boundary condition directly into the system of linear equations.
def apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter):

    Ks = pars['Ks']

    # Extract current h0 and h1 (first subsurface node)
    h0 = h_current[0]
    h1 = h_current[1]
    if h0 >= 0:
        flux = min(R, Ks)
        h_current[0] = 0.0
    else:
        dhdz = (h1 - h0)/dz
        Ktop = K_current[0]

        infil_cap = -Ktop * (dhdz + 1)
        if R <= infil_cap:
            flux = R
        else:
            #h_current[0] = 0
            dhdz = (h_current[1] - h0)/dz
            infil_cap = -Ks * (dhdz + 1)
            flux = min(R, infil_cap)

    flux = max(0, flux)
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


def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_top, K_bottom):
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

    # z increases downward, so i + 1/2 is upper layer and i-1/2 is the lower layer
    # top boundary in case of rainfall
    A[0, 0] = C_val[0] / dt + (K_half[0]) / dz ** 2
    A[0, 1] = - K_half[0] / dz ** 2
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt - K_half[0] / dz
    # middle compartments

    for i in range(1, n):
        A[i, i - 1] = - K_half[i - 1] / dz ** 2
        A[i, i] = C_val[i] / dt + (K_half[i] + K_half[i - 1]) / dz ** 2
        A[i, i + 1] = - K_half[i] / dz ** 2
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i-1] - K_half[i]) / dz

    # bottom boundary
    if h_curr[n] > 0:
        A[n, n-1] = 0.0
        A[n, n] = 1.0
        b[n] = 0.0
    else:
        A[n, n - 1] = - K_half[n - 1] / dz ** 2
        A[n, n] = C_val[n] / dt + ( K_half[n - 1]) / (dz) ** 2

        # Let bottom boundary be 0 for now
        # b[n] = 0
        # free drainage
        b[n] = (theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + (K_half[n-1] - K_bottom)/dz

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
    pars['Ks'] = 0.03 # m/hr
    pars['neta'] = 0.5 # fixed value
    pars['Ss'] = 0.000001

    dz = 0.1
    Nz = 10
    #T = 100
    dt = 1
    L = 1
    z = np.linspace(0, L, Nz)
    #time = np.arange(0, T + dt, dt)
    Nt = 24

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
        rainfall = simulate_rainfall(step)
        irrigation = simulate_irrigation(step)
        R = rainfall + irrigation
        if step > 0:
            theta_prev = theta_val[:, step - 1]
            h_current = psi[:, step - 1].copy()

        else:
            h_current = np.full(Nz, -60.5)
            h_current[0] = -30.0
            theta_prev = thetaf(h_current + 1.0, pars)


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

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, dz, dt, K_step[0], K_step[-1])

            # if R != 0:
            #     runoff = apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter)
            # else:
            #     runoff = 0.0
            flux, runoff = calculate_top_flux_runoff(R, h_current, K_current, dz, pars)
            b[0] += (-flux)/dz # downward positive flux

            h_new = spsolve(A, b)
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance
            max_diff = np.max(np.abs(h_new - h_current))
            if max_diff < tolerance:
                break

            # relaxation factor
            if np.all(h_current >= -1e-6):
                h_current = h_new
            else:
                h_current = h_new * 0.6 + h_current * 0.4

            # if soil becomes saturated, it doesn't make sense to calculate it.
            # h_current[h_current > 0] = 0.0

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