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


def psi_fun(theta, pars):
    '''Compute matric potential from the volumetric water content'''
    pw1 = 1/pars['m']
    denom = theta - pars['thetaR']
    denom[denom < 0.0] = 0.001
    Se = (pars['thetaS'] - pars['thetaR'])/denom
    temp1 = Se**pw1 - 1
    return np.array(-(1.0 /pars['alpha']) * (temp1 ** (1/pars['n'])))

def C(psi, pars):
    "Calculate specific moisture capacity C(h) = d(theta)/dh"
    nume = pars['m'] * pars['n'] * pars['alpha'] * (pars['alpha'] * np.abs(psi))**(pars['n']-1)
    denom = (1 + (pars['alpha'] * np.abs(psi))**(pars['n']))**(pars['m']+1)
    return np.array(-(pars['thetaS'] - pars['thetaR']) * nume/denom * psi/(np.abs(psi)+1e-10))

# def K(psi,pars):
#     '''Compute hydraulic conductivity via the Mualem model'''
#     Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
#     Se[psi>0.]=1.0
#     return np.array(pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2)

def K(psi,pars):
    alpha = pars['alpha']
    n = pars['n']
    m = pars['m']
    Se = (1 + (-psi*alpha)**n)**(-m)
    nume = (1 - (alpha * (-psi))**(n-1)* Se )**2
    denom = (1 + (alpha * (-psi))**n )**(m/2)
    val = (nume/denom) * pars['Ks']
    return val

max_iter = 50  # maximum possible picard iterations




# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/WR011i001p00102
def calculate_top_flux_infiltration(R, h_current, pars, dz, Ft, theta_val):
    """
    Based on Green Ampt Equation
    Args:
        R: rainfall
        h_current: matric potential of soil layers
        dz: depth of first layer
        pars: user defined parameters
    Returns:
        runoff: amount of water that runoffs the surface
        infl: flux that enters the soil surface for the timestep
    """

    Ks = pars['Ks'][0]
    del_theta = pars['thetaS'][0] - theta_val
    h0 = h_current[0] # measured at center of top compartment
    # zc = dz/2.0 # 50cm
    if Ft == 0:
        Ft = 1e-6
    ft = Ks * (1 - h0 * del_theta / Ft)
    rc = max(0, ft)
    if R < rc:
        infiltration = R
    else:
        infiltration = rc

    return min(R, infiltration), max(0, R - infiltration)


def compute_deep_percolation(K_current, dt):
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
    q_bottom = K_current[-1] #(m/hr)

    # Convert flux to volume per unit area over Δt (e.g., meters of water)
    percolation = q_bottom * dt # dt

    return q_bottom, percolation #(m)



# mean is used to calculate the conductivity on the surface
def mean(K_i, K_j, dz):
    weighted_mean = (K_i * (dz[1:]/2) + K_j * dz[:-1]/2) / ((dz[1:]+dz[:-1])*0.5)
    return weighted_mean

def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_top, K_bottom, diff_water, iter=0, R=0):
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
    dist_iplus1_i = (dz[0] + dz[1]) / 2.0
    A[0, 0] = C_val[0] / dt + (K_half[0]) / (dist_iplus1_i * dz[0])
    A[0, 1] = - K_half[0] / (dist_iplus1_i * dz[0])
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt - K_half[0] / dz[0] + diff_water[0]
    # middle compartments

    for i in range(1, n):
        dist_iminus1_i = (dz[i-1] + dz[i])/2.0
        A[i, i - 1] = - K_half[i - 1] / (dist_iminus1_i * dz[i])
        dist_iplus1_i = (dz[i] + dz[i+1]) / 2.0
        A[i, i] = C_val[i] / dt + K_half[i]/(dist_iplus1_i * dz[i]) + K_half[i - 1] / (dist_iminus1_i * dz[i])
        A[i, i + 1] = - K_half[i] / (dist_iplus1_i * dz[i])
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i-1] - K_half[i]) / dz[i]
        if R == 0 or iter > 10:
            b[i] += diff_water[i]

    # bottom boundary
    if h_curr[n] > 0:
        A[n, n-1] = 0.0
        A[n, n] = 1.0
        b[n] = 0.0
    else:
        last_dz = dz.iloc[-1]
        dist_iminus1_i = (dz.iloc[-2] + last_dz) / 2.0
        A[n, n - 1] = - K_half[n - 1] / (last_dz * dist_iminus1_i)
        A[n, n] = C_val[n] / dt + (K_half[n - 1]) / (last_dz * dist_iminus1_i)

        # Let bottom boundary be 0 for now
        # b[n] = 0
        # free drainage
        b[n] = ((theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + (K_half[n-1] - K_bottom)/last_dz)
        if R == 0 or iter > 10:
            b[n] += diff_water[-1]

    return A, b



class RichardEquationSolver:
    def __init__(self, soil_profile, prev_cond, time_step='d'):
        self.pars = {}
        self.pars['thetaR'] = soil_profile.th_r
        self.pars['thetaS'] = soil_profile.th_s
        self.pars['alpha'] = soil_profile.alpha
        self.pars['n'] = soil_profile.n_param
        self.pars['m'] = 1 - 1 / self.pars['n']
        self.time_step = time_step.lower()
        self.soil_profile = soil_profile
        self.prev_cond = prev_cond
        if self.time_step == 's':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 60 * 60) # m/sec
            self.Nt = 60
        elif self.time_step == 'm':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 60) # m/minute
            self.Nt = 60
        elif self.time_step == 'h':
            self.pars['Ks'] = (soil_profile.Ksat / 1000)/24.0 # m/hr
            self.Nt = 24
        else:
            self.pars['Ks'] = (soil_profile.Ksat / 1000) # m/day
            self.Nt = 1

        self.pars['neta'] = soil_profile.neta  # fixed value
        self.dz = soil_profile.dz
        self.Nz = len(self.dz)
        # T = 100
        self.dt = 1
        self.L = sum(self.dz)
        # z = np.linspace(0, L, Nz)
        # time = np.arange(0, T + dt, dt)
        self.psi = np.zeros((self.Nz, self.Nt))
        self.psi[:, 0] = psi_fun(prev_cond.th, self.pars)
        self.theta_val = np.zeros((self.Nz, self.Nt))
        self.runoff_val = np.zeros(self.Nt)
        self.deep_percolation_val = np.zeros(self.Nt)
        self.K_val = np.zeros((self.Nz, self.Nt))
        self.C_val = np.zeros((self.Nz, self.Nt))
        self.tolerance = 1e-10
        self.prev_cond = prev_cond
        self.Infiltration_So_Far = 0.0


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
        converged = False
        # at each timestep, update the value of theta, K and C for each compartment
        for iter in range(max_iter):
            # at each step of picard iteration
            # values at iter = m
            self.theta_val[:, step] = thetaf(h_current, self.pars)
            self.K_val[:, step] = K_current
            self.C_val[:, step] = C_current

            # first layer is bit complicated as we need to set up boundary condition.
            K_half = mean(K_current[:-1].values, K_current[1:].values, self.dz.values)
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt, K_current[0],
                                   K_current.values[-1], diff_water_content)

            # if R != 0:
            #     runoff = apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter)
            # else:
            #     runoff = 0.0
            flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0], self.Infiltration_So_Far, theta_current[0])
            b[0] += (flux / self.dz[0])  # downward positive flux
            Infl = flux * self.dt
            h_new = spsolve(csc_matrix(A), b)
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance
            max_diff = np.max(np.abs(h_new - h_current))
            if max_diff < self.tolerance:
                converged = True

            if max(h_new) > -1e-10:
                factor = 0.95
                h_current = h_new #* factor + h_current * (1 - factor)
            else:
                h_current = h_new

            if max(h_current) >= 0.0:
                if self.time_step == 'm':
                    h_current[h_current > 0.0] = 0.0
                    print('No Convergence')
                    converged = True
                else:
                    break

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)
            if converged:
                break

        # update theta/K/C

        deep_percolation = 0.0
        if self.time_step != 's' and not converged:
            runoff = 0.0
            Infl = 0.0
            if self.time_step == 'h':
                solver = RichardEquationSolver(self.soil_profile, new_cond, time_step='m')
            elif self.time_step == 'm':
                solver = RichardEquationSolver(self.soil_profile, new_cond, time_step='s')
            for _step in range(solver.Nt):
                converged, theta_current, _deep_perc, _runoff, _infl, K_current, C_current, h_current = solver.solve(_step, new_cond, irrigation/60.0, rainfall/60.0)
                new_cond.th = theta_current
                runoff += _runoff
                Infl += _infl
                deep_percolation += _deep_perc
        self.Infiltration_So_Far += Infl
        self.theta_val[:, step] = theta_current
        self.K_val[:, step] = K_current
        self.C_val[:, step] = C_current
        self.runoff_val[step] = runoff  # unit: m/hr
        self.psi[:, step] = h_current
        # calculate deep percolation
        if converged:
            q_bottom, deep_percolation = compute_deep_percolation(K_current.values, self.dt)
        if not converged:
            runoff += R
        self.deep_percolation_val[step] = deep_percolation

        new_theta = self.theta_val[:, step].flatten()
        return converged, new_theta, deep_percolation, runoff, Infl, K_current, C_current, h_current


    def solve_daily(self, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation)/1000.0

        theta_prev = np.array(self.prev_cond.th)
        h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        Infl = 0.0
        diff_water_content = new_cond.th - theta_prev
        converged = False
        # at each timestep, update the value of theta, K and C for each compartment
        for iter in range(max_iter):
            # at each step of picard iteration
            # values at iter = m
            # self.theta_val[:, step] = thetaf(h_current, self.pars)
            # self.K_val[:, step] = K(h_current, self.pars)
            # self.C_val[:, step] = C(h_current, self.pars)

            #K_step = self.K_val[:, step]
            # first layer is bit complicated as we need to set up boundary condition.
            K_half = mean(K_current[:-1].values, K_current[1:].values, self.dz.values)
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt, K_current[0],
                                   K_current.values[-1], diff_water_content, iter, R)

            # if R != 0:
            #     runoff = apply_top_bc(A, b, R, h_current, K_current, dz, pars, iter)
            # else:
            #     runoff = 0.0

            flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0], self.Infiltration_So_Far, theta_current[0])
            b[0] += (flux / self.dz[0])  # downward positive flux
            Infl = flux * self.dt
            h_new = spsolve(csc_matrix(A), b)
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance
            max_diff = np.max(np.abs(h_new - h_current))
            if max_diff < self.tolerance:
                h_current = h_new
                converged = True
                break

            # relaxation factor
            # if h_new[0] < 0.0:
            #     h_current = h_new
            # else:
                # if soil becomes saturated, it doesn't make sense to calculate it.
            if max(h_new) > -1e-10:
                factor = 0.6
                h_current = h_new * factor + h_current * (1 - factor)
            else:
                h_current = h_new

            if max(h_current) > 0.0:
                break

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

        # calculate deep percolation
        q_bottom, deep_percolation = compute_deep_percolation(K_current.values, self.dt)
        #self.deep_percolation_val[step] = deep_percolation

        new_theta = theta_current.flatten()#self.theta_val[:, step].flatten()
        return converged, new_theta, deep_percolation, runoff, Infl, K_current