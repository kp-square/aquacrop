import math

import numpy as np
from typing import Tuple, TYPE_CHECKING
from scipy.sparse import csc_matrix
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
    return np.array(pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se)


def C(psi, pars):
    "Calculate specific moisture capacity C(h) = d(theta)/dh"
    nume = pars['m'] * pars['n'] * pars['alpha'] * (pars['alpha'] * np.abs(psi))**(pars['n']-1)
    denom = (1 + (pars['alpha'] * np.abs(psi))**(pars['n']))**(pars['m']+1)
    return np.array(-(pars['thetaS'] - pars['thetaR']) * nume/denom * psi/(np.abs(psi)+1e-10))

def K(psi,pars):
    '''Compute hydraulic conductivity via the Mualem model'''
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Se[psi>0.]=1.0
    return np.array(pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2)

def psi_fun(theta, pars):
    '''Compute matric potential from the volumetric water content'''
    pw1 = pars['n']/(pars['n']-1)
    temp1 = ((pars['thetaS'] - pars['thetaR'])/(theta - pars['thetaR']))**pw1 - 1
    return np.array(-(1.0 /pars['alpha']) * (temp1 ** (1/pars['n'])))


max_iter = 100  # maximum possible picard iterations


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
    grad_H = (h1 - 0.0) / dz + 1
    q_soil_potential = - K0 * grad_H

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
    q_bottom = K_half[-1] * 1.0 #(/hr)

    # Convert flux to volume per unit area over Δt (e.g., meters of water)
    percolation = q_bottom * dz

    return q_bottom, percolation #(m/hr)


def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_top, K_bottom, diff_water):
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
    A[0, 0] = C_val[0] / dt + (K_half[0]) / dz[0] ** 2
    A[0, 1] = - K_half[0] / dz[0] ** 2
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt - K_half[0] / dz[0] + diff_water[0]
    # middle compartments

    for i in range(1, n):
        A[i, i - 1] = - K_half[i - 1] / dz[i] ** 2
        A[i, i] = C_val[i] / dt + (K_half[i] + K_half[i - 1]) / dz[i] ** 2
        A[i, i + 1] = - K_half[i] / dz[i] ** 2
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i-1] - K_half[i]) / dz[i] + diff_water[i]

    # bottom boundary
    if h_curr[n] > 0:
        A[n, n-1] = 0.0
        A[n, n] = 1.0
        b[n] = 0.0
    else:
        last_dz = dz.iloc[-1]
        A[n, n - 1] = - K_half[n - 1] / last_dz ** 2
        A[n, n] = C_val[n] / dt + ( K_half[n - 1]) / (last_dz) ** 2

        # Let bottom boundary be 0 for now
        # b[n] = 0
        # free drainage
        b[n] = (theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + (K_half[n-1] - K_bottom)/last_dz + diff_water[-1]

    return A, b



class RichardEquationSolver:
    def __init__(self, soil_profile, prev_cond):
        self.pars = {}
        self.pars['thetaR'] = soil_profile.th_r
        self.pars['thetaS'] = soil_profile.th_s
        self.pars['alpha'] = soil_profile.alpha
        self.pars['n'] = soil_profile.n_param
        self.pars['m'] = 1 - 1 / self.pars['n']
        self.pars['Ks'] = (soil_profile.Ksat / 1000)/24.0 # m/hr
        self.pars['neta'] = soil_profile.neta  # fixed value
        self.dz = soil_profile.dz
        self.Nz = len(self.dz)
        # T = 100
        self.dt = 1
        self.L = sum(self.dz)
        # z = np.linspace(0, L, Nz)
        # time = np.arange(0, T + dt, dt)
        self.Nt = 24
        self.psi = np.zeros((self.Nz, self.Nt))
        self.psi[:, 0] = psi_fun(prev_cond.th, self.pars)
        self.theta_val = np.zeros((self.Nz, self.Nt))
        self.runoff_val = np.zeros(self.Nt)
        self.deep_percolation_val = np.zeros(self.Nt)
        self.K_val = np.zeros((self.Nz, self.Nt))
        self.C_val = np.zeros((self.Nz, self.Nt))
        self.tolerance = 1e-5
        self.prev_cond = prev_cond

    def solve(self, step, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation)/1000.0
        if step > 0:
            theta_prev = self.theta_val[:, step - 1]
            h_current = self.psi[:, step - 1].copy()
        else:
            theta_prev = np.array(self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        Infl = 0.0
        diff_water_content = new_cond.th - theta_prev
        # at each timestep, update the value of theta, K and C for each compartment
        for iter in range(max_iter):
            # at each step of picard iteration
            # values at iter = m
            self.theta_val[:, step] = thetaf(h_current, self.pars)
            self.K_val[:, step] = K(h_current, self.pars)
            self.C_val[:, step] = C(h_current, self.pars)

            K_step = self.K_val[:, step]
            # first layer is bit complicated as we need to set up boundary condition.
            K_half = mean(K_step[:-1], K_step[1:])
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt, K_step[0],
                                   K_step[-1], diff_water_content)

            # if R != 0:
            #     runoff = apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter)
            # else:
            #     runoff = 0.0
            flux, runoff = calculate_top_flux_runoff(R, h_current, K_current, self.dz[0], self.pars)
            b[0] += (-flux) / self.dz[0]  # downward positive flux
            Infl = flux * self.dz[0]
            h_new = spsolve(csc_matrix(A), b)
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance
            max_diff = np.max(np.abs(h_new - h_current))
            if max_diff < self.tolerance:
                break

            # relaxation factor
            if np.all(h_current >= -1e-6):
                h_current = h_new
            else:
                h_current = h_new * 0.6 + h_current * 0.4

            # if soil becomes saturated, it doesn't make sense to calculate it.
            # h_current[h_current > 0] = 0.0

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

        # update theta/K/C
        self.theta_val[:, step] = theta_current
        self.K_val[:, step] = K_current
        self.C_val[:, step] = C_current
        self.runoff_val[step] = runoff  # unit: m/hr
        self.psi[:, step] = h_current

        # calculate deep percolation
        h_bottom = h_current[-1]
        q_bottom, deep_percolation = compute_deep_percolation(K_half, h_bottom, h_current, self.dz.iloc[-1], self.dt)
        self.deep_percolation_val[step] = deep_percolation

        new_cond.th = self.theta_val[:, step].flatten()
        return new_cond, deep_percolation, runoff, Infl, K_current