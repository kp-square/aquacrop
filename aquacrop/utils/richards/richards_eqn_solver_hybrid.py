import copy
import math

import numpy as np
import pandas as pd
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

def K(psi,pars):
    '''Compute hydraulic conductivity via the Mualem model'''
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Se[psi>0.]=1.0
    k_val = np.array(pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2)
    return np.maximum(k_val, 1e-30)

# def K(psi,pars):
#     alpha = pars['alpha']
#     n = pars['n']
#     m = pars['m']
#     Se = (1 + (-psi*alpha)**n)**(-m)
#     nume = (1 - (alpha * (-psi))**(n-1)* Se )**2
#     denom = (1 + (alpha * (-psi))**n )**(m/2)
#     val = (nume/denom) * pars['Ks']
#     return val


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

def wieghted_geometric_mean(K_i, K_j, dz):
    dist_i = dz[:-1]/2.0
    dist_j = dz[1:]/2.0
    total_dist = dist_i + dist_j
    log_K_i = np.log(K_i)
    log_K_j = np.log(K_j)
    weighted_log_mean = (log_K_i * dist_j + log_K_j * dist_i) / total_dist
    K_half = np.exp(weighted_log_mean)
    return K_half

def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_bottom):
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
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt - K_half[0] / dz[0]
    # middle compartments

    for i in range(1, n):
        dist_iminus1_i = (dz[i-1] + dz[i])/2.0
        A[i, i - 1] = - K_half[i - 1] / (dist_iminus1_i * dz[i])
        dist_iplus1_i = (dz[i] + dz[i+1]) / 2.0
        A[i, i] = C_val[i] / dt + K_half[i]/(dist_iplus1_i * dz[i]) + K_half[i - 1] / (dist_iminus1_i * dz[i])
        A[i, i + 1] = - K_half[i] / (dist_iplus1_i * dz[i])
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i-1] - K_half[i]) / dz[i]

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

    return A, b



class RichardEquationSolver:
    def __init__(self, soil_profile, prev_cond, time_step='d'):
        self.pars = {}
        self.pars['thetaR'] = soil_profile.th_r
        self.pars['thetaS'] = soil_profile.th_s
        self.pars['alpha'] = soil_profile.alpha
        self.pars['n'] = soil_profile.n_param
        self.pars['m'] = 1 - 1 / self.pars['n']
        self.pars['neta'] = soil_profile.neta
        self.time_step = time_step.lower()
        self.soil_profile = soil_profile
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
        self.tolerance = 1e-5
        self.prev_cond = copy.deepcopy(prev_cond)
        self.Infiltration_So_Far = 0.0
        self.max_iter = 20

    def solve(self, step, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation) / 1000.0
        # If soil is super dry, i.e. in any layer moisture is below theta_r don't proceed
        self.tolerance = 1e-5

        SATURATION_THRESHOLD_h = -0.03
        h_eff_sat = np.full(self.Nz, SATURATION_THRESHOLD_h)
        theta_eff_sat = thetaf(h_eff_sat, self.pars)
        infiltration_source = np.zeros(self.Nz)

        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)

        root_water_uptake = (new_cond.th - theta_prev_prev) / self.dt

        Infl, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz,
                                                       self.Infiltration_So_Far, theta_prev, theta_eff_sat)
        top_flux = Infl

        diff_water_content = root_water_uptake

        dry_soil = False
        no_deep_perc = False
        if any(np.array(self.prev_cond.th) < (self.soil_profile.th_r + 1e-3)):
            dry_soil = True
            epsilon = 1e-3
            theta_prev = np.maximum(self.pars['thetaR'] + epsilon, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        converged = False
        prev_max_diff = 1.0
        # at each timestep, update the value of theta, K and C for each compartment

        # Anderson Acceleration memory size
        m_aa = 5
        h_hist = np.zeros((len(h_current), m_aa))
        f_hist = np.zeros((len(h_current), m_aa))
        beta = 1.0
        divergence_count = 0
        for iter in range(self.max_iter):

            K_half = wieghted_geometric_mean(K_current[:-1], K_current[1:], self.dz.values)

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                   K_current[0],
                                   K_current[-1], diff_water_content, 0, 0)

            b[0] += (top_flux / self.dz[0])  # downward positive flux

            try:
                b_mpi = b - (A @ h_current)
                delta_h = spsolve(csc_matrix(A), b_mpi)
            except:
                converged = False
                break

            max_diff = np.linalg.norm(delta_h)

            if max_diff < self.tolerance:
                converged = True
                break

            max_diff_differential = (max_diff - prev_max_diff)
            if max_diff_differential > 0:
                divergence_count += 1
            else:
                divergence_count -= 1
            prev_max_diff = max_diff

            if max(h_current + delta_h) > -1e-3:
                beta = 0.5
            elif max(h_current) > -0.1:
                if max_diff_differential > -0.25:
                    beta = 0.7
            elif divergence_count > 3:
                beta = 0.5
            elif max_diff > 4.0:
                beta = 0.25
            elif max_diff > 2.0:
                beta = 0.5
            elif max_diff > 1.0:
                beta = 0.75
            else:
                beta = 1.0

            k_aa = iter % m_aa
            h_hist[:, k_aa] = h_current
            f_hist[:, k_aa] = delta_h
            m_eff = min(iter + 1, m_aa)
            if iter == 0:
                h_new = h_current + beta * delta_h
            else:
                f_diff = f_hist[:, 1:m_eff] - f_hist[:, 0:m_eff - 1]
                gamma, _, _, _ = np.linalg.lstsq(f_diff, -delta_h, rcond=None)
                h_update_hist = h_hist[:, 1:m_eff] - h_hist[:, 0:m_eff - 1]
                h_new = (h_current + delta_h) - np.dot(h_update_hist, gamma) - np.dot(f_diff, gamma)

                if beta != 1.0:
                    h_new = h_current + beta * (h_new - h_current)

            h_current = h_new

            if max_diff_differential > 5.0 and self.time_step != 'm':
                break

            if np.any(h_current >= 0.0):
                break

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

            if converged or divergence_count > 10:
                break

        # update theta/K/C

        if iter > 20 and max_diff < 0.001:
            converged = True

        deep_percolation = 0.0
        fallback_used = False
        if not converged and self.time_step != 'm':
            fallback_used = True
            solver = None
            runoff = 0.0
            Infl = 0.0
            if self.time_step == 'h':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='sm')
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if self.time_step == 'sm':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='m')
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if solver:
                dry_soil = False
                diff_water_sub = root_water_uptake / solver.Nt
                sub_step_theta_prev = theta_prev_prev.copy()
                new_cond_sub = copy.deepcopy(new_cond)

                for _step in range(solver.Nt):
                    new_cond_sub.th = sub_step_theta_prev + diff_water_sub

                    converged, theta_current, _deep_perc, _runoff, _infl = solver.solve(
                        step * self.Nt + _step, new_cond_sub, irrigation / solver.Nt, rainfall / solver.Nt)

                    sub_step_theta_prev = theta_current.copy()

                    runoff += _runoff
                    Infl += _infl
                    deep_percolation += _deep_perc
        elif not converged:
            dry_soil = False
            theta_current, deep_percolation, runoff, Infl = self.handle_non_convergence(R, theta_prev_prev,
                                                                                        root_water_uptake)

        self.Infiltration_So_Far += Infl
        # calculate deep percolation
        if converged:
            if dry_soil:
                # dry layers
                mask = np.array(theta_prev_prev < (self.soil_profile.th_r + 1e-3))
                for i, val in enumerate(mask):
                    if val:
                        theta_current[i] = theta_prev_prev[i] + theta_current[i] - ref_prev_theta_current[i]
            if not fallback_used and not no_deep_perc:
                q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
            prev_water = sum(self.prev_cond.th * self.dz) * 1000
            new_water = sum(theta_current * self.dz) * 1000
            deep_perc = deep_percolation * 1000
            evap_traspir = sum(root_water_uptake * self.dz) * 1000
            infilt = Infl * 1000
            err = prev_water - new_water + infilt - deep_perc + evap_traspir
            if abs(err) > 0.02:
                print('error')

        self.prev_cond.th = theta_current

        new_theta = theta_current.flatten()
        return converged, new_theta, deep_percolation, runoff, Infl


    def solve_daily(self, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation) / 1000.0
        # If soil is super dry, i.e. in any layer moisture is below theta_r don't proceed
        self.tolerance = 1e-5

        SATURATION_THRESHOLD_h = -0.03
        h_eff_sat = np.full(self.Nz, SATURATION_THRESHOLD_h)
        theta_eff_sat = thetaf(h_eff_sat, self.pars)

        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)

        root_water_uptake = (new_cond.th - theta_prev_prev) / self.dt

        Infl, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz,
                                                       self.Infiltration_So_Far, theta_prev, theta_eff_sat)
        top_flux = Infl

        diff_water_content = root_water_uptake

        dry_soil = False
        no_deep_perc = False
        if any(np.array(self.prev_cond.th) < (self.soil_profile.th_r + 1e-3)):
            dry_soil = True
            epsilon = 1e-3
            theta_prev = np.maximum(self.pars['thetaR'] + epsilon, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        converged = False
        prev_max_diff = 1.0
        # at each timestep, update the value of theta, K and C for each compartment

        # Anderson Acceleration memory size
        m_aa = 5
        h_hist = np.zeros((len(h_current), m_aa))
        f_hist = np.zeros((len(h_current), m_aa))
        beta = 1.0
        divergence_count = 0
        for iter in range(self.max_iter):

            K_half = wieghted_geometric_mean(K_current[:-1], K_current[1:], self.dz.values)

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                   K_current[0],
                                   K_current[-1], diff_water_content, 0, 0)

            b[0] += (top_flux / self.dz[0])  # downward positive flux

            try:
                b_mpi = b - (A @ h_current)
                delta_h = spsolve(csc_matrix(A), b_mpi)
            except:
                converged = False
                break

            max_diff = np.linalg.norm(delta_h)

            if max_diff < self.tolerance:
                converged = True
                break

            max_diff_differential = (max_diff - prev_max_diff)
            if max_diff_differential > 0:
                divergence_count += 1
            else:
                divergence_count -= 1
            prev_max_diff = max_diff

            if max(h_current + delta_h) > -1e-3:
                beta = 0.5
            elif max(h_current) > -0.1:
                if max_diff_differential > -0.25:
                    beta = 0.7
            elif divergence_count > 3:
                beta = 0.5
            elif max_diff > 4.0:
                beta = 0.25
            elif max_diff > 2.0:
                beta = 0.5
            elif max_diff > 1.0:
                beta = 0.75
            else:
                beta = 1.0

            k_aa = iter % m_aa
            h_hist[:, k_aa] = h_current
            f_hist[:, k_aa] = delta_h
            m_eff = min(iter + 1, m_aa)
            if iter == 0:
                h_new = h_current + beta * delta_h
            else:
                f_diff = f_hist[:, 1:m_eff] - f_hist[:, 0:m_eff - 1]
                gamma, _, _, _ = np.linalg.lstsq(f_diff, -delta_h, rcond=None)
                h_update_hist = h_hist[:, 1:m_eff] - h_hist[:, 0:m_eff - 1]
                h_new = (h_current + delta_h) - np.dot(h_update_hist, gamma) - np.dot(f_diff, gamma)

                if beta != 1.0:
                    h_new = h_current + beta * (h_new - h_current)

            h_current = h_new

            if max_diff_differential > 5.0 and self.time_step != 'm':
                break

            if np.any(h_current >= 0.0):
                break

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

            if converged or divergence_count > 10:
                break

        # update theta/K/C

        if iter > 20 and max_diff < 0.001:
            converged = True

        deep_percolation = 0.0
        fallback_used = False
        if not converged and self.time_step != 'm':
            fallback_used = True
            solver = None
            runoff = 0.0
            Infl = 0.0
            if self.time_step == 'h':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='sm')
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if self.time_step == 'sm':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='m')
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if solver:
                dry_soil = False
                diff_water_sub = root_water_uptake / solver.Nt
                sub_step_theta_prev = theta_prev_prev.copy()
                new_cond_sub = copy.deepcopy(new_cond)

                for _step in range(solver.Nt):
                    new_cond_sub.th = sub_step_theta_prev + diff_water_sub

                    converged, theta_current, _deep_perc, _runoff, _infl = solver.solve(
                        step * self.Nt + _step, new_cond_sub, irrigation / solver.Nt, rainfall / solver.Nt)

                    sub_step_theta_prev = theta_current.copy()

                    runoff += _runoff
                    Infl += _infl
                    deep_percolation += _deep_perc
        elif not converged:
            dry_soil = False
            theta_current, deep_percolation, runoff, Infl = self.handle_non_convergence(R, theta_prev_prev,
                                                                                        root_water_uptake)

        self.Infiltration_So_Far += Infl
        # calculate deep percolation
        if converged:
            if dry_soil:
                # dry layers
                mask = np.array(theta_prev_prev < (self.soil_profile.th_r + 1e-3))
                for i, val in enumerate(mask):
                    if val:
                        theta_current[i] = theta_prev_prev[i] + theta_current[i] - ref_prev_theta_current[i]
            if not fallback_used and not no_deep_perc:
                q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
            prev_water = sum(self.prev_cond.th * self.dz) * 1000
            new_water = sum(theta_current * self.dz) * 1000
            deep_perc = deep_percolation * 1000
            evap_traspir = sum(root_water_uptake * self.dz) * 1000
            infilt = Infl * 1000
            err = prev_water - new_water + infilt - deep_perc + evap_traspir
            if abs(err) > 0.02:
                print('error')

        self.prev_cond.th = theta_current

        new_theta = theta_current.flatten()
        return converged, new_theta, deep_percolation, runoff, Infl