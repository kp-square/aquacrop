import math

import numpy as np
from typing import Tuple, TYPE_CHECKING
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import copy

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
    cval = np.array(-(pars['thetaS'] - pars['thetaR']) * nume/denom * psi/(np.abs(psi)+1e-10))
    return cval

def K(psi,pars):
    '''Compute hydraulic conductivity via the Mualem model'''
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    Se[psi>0.]=1.0
    k_val = np.array(pars['Ks']*Se**pars['neta']*(1-(1-Se**(1/pars['m']))**pars['m'])**2)
    return np.maximum(k_val, 1e-40)

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


def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_top, K_bottom, diff_water, iter, R):
    '''
    K_half : Value of K at boundary of each soil compartments
    C_val: value of C at each compartment
    theta_val: value of theta at each compartment
    h_current: value of h at each compartment
    dz = depth of one compartment
    dt = timestep

    ***Note**
    Make sure iter = 0 and R = 0 when this function is called by hourly or lesser solver, they
    should be passed on daily solver only.

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
        self.prev_cond = copy.deepcopy(prev_cond)
        self.max_iter = 50
        if self.time_step == 's':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 60 * 60) # m/sec
            self.Nt = 60

        elif self.time_step == 'm':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 12 * 5) # m/minute
            self.Nt = 5

        elif self.time_step == 'sm':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 12) # m/5 minute
            self.Nt = 12

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
        self.psi = np.zeros((self.Nz, self.Nt))
        self.theta_val = np.zeros((self.Nz, self.Nt))
        self.runoff_val = np.zeros(self.Nt)
        self.deep_percolation_val = np.zeros(self.Nt)
        self.tolerance = 1e-5
        self.Infiltration_So_Far = 0.0

    def solve(self, step, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation)/1000.0
        # If soil is super dry, i.e. in any layer moisture is below theta_r don't proceed
        self.tolerance = 1e-5

        SATURATION_THRESHOLD_h = -0.05
        h_eff_sat = np.full(self.Nz, SATURATION_THRESHOLD_h)
        theta_eff_sat = thetaf(h_eff_sat, self.pars)
        infiltration_source = np.zeros(self.Nz)

        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)


        diff_water_content = (new_cond.th - theta_prev_prev)/self.dt
        runoff = 0.0
        Infl = 0.0
        top_flux = 0.0
        if R > 0 and h_current[0] >= SATURATION_THRESHOLD_h:
            available_volume = R
            for i in range(self.Nz):
                if available_volume <= 1e-9:
                    break
                volume_to_fill = max(0, (theta_eff_sat[i] - theta_prev[i]) * self.dz[i])
                water_to_add = min(available_volume, volume_to_fill)
                infiltration_source[i] = water_to_add / self.dz[i] / self.dt
                available_volume -= water_to_add
            runoff = available_volume
            Infl = R - runoff
            top_flux = 0.0
        else:
            Infl, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0],
                                                           self.Infiltration_So_Far, theta_prev[0])
            top_flux = Infl

        diff_water_content = diff_water_content + infiltration_source

        dry_soil = False
        no_deep_perc = False
        if any(np.array(self.prev_cond.th) < (self.soil_profile.th_r + 1e-3)):
            dry_soil = True
            # self.tolerance = 1e-3
            epsilon = 1e-3
            theta_prev = np.maximum(self.pars['thetaR'] + epsilon, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)
            #self.prev_cond.th = new_cond.th

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        #Infl = 0.0
        #runoff = 0.0
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
            # at each step of picard iteration
            # values at iter = m

            # first layer is bit complicated as we need to set up boundary condition.
            #K_half = experimental_mean(self.dz, h_current, K_current)
            K_half = wieghted_geometric_mean(K_current[:-1], K_current[1:], self.dz.values)
            if max(h_current[1:]) > SATURATION_THRESHOLD_h: # matric potential is less than 5 cm
                # if middle layer is very wet and near saturation
                K_half = mean(K_current[:-1], K_current[1:], self.dz.values)
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt, K_current[0],
                                   K_current[-1], diff_water_content, 0, 0)


            #flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0], self.Infiltration_So_Far, theta_current[0])
            b[0] += (top_flux / self.dz[0])  # downward positive flux
            #Infl = top_flux * self.dt

            try:
                b_mpi = b - (A @ h_current)
                delta_h = spsolve(csc_matrix(A), b_mpi)
            except:
                converged = False
                break
            # Check convergence
            # if np.max(np.abs(h_new - h_current)) < tolerance:
            #     break
            # convergence check with relative tolerance

            # max_diff = np.max(np.abs(delta_h))

            max_diff = np.linalg.norm(delta_h)

            if max_diff < self.tolerance:
                converged = True
                break

            max_diff_differential = (max_diff - prev_max_diff)
            if max_diff_differential > 0:
                divergence_count += 1
            prev_max_diff = max_diff

            if max(h_current + delta_h) > -1e-3:
                beta = 0.5

            elif max(h_current) > -0.1:
                if max_diff_differential > -0.25:
                    beta = 0.7

            elif divergence_count > 3:
                beta = 0.5
            else:
                beta = 1.0

            k_aa = iter % m_aa
            h_hist[:, k_aa] = h_current
            f_hist[:, k_aa] = delta_h
            m_eff = min(iter+1, m_aa)
            if iter == 0:
                h_new = h_current + beta * delta_h
            else:
                f_diff = f_hist[:, 1:m_eff] - f_hist[:, 0:m_eff-1]
                gamma, _, _, _ = np.linalg.lstsq(f_diff, -delta_h, rcond=None)
                h_update_hist = h_hist[:, 1:m_eff] - h_hist[:, 0:m_eff-1]
                h_new = (h_current + delta_h) - np.dot(h_update_hist, gamma) - np.dot(f_diff, gamma)

                if beta != 1.0:
                    h_new = h_current + beta * (h_new - h_current)

            #h_new = h_current + delta_h
            h_current = h_new

            if max_diff > 1.0 and max_diff_differential > 5.0 and self.time_step != 'm':
                converged = False
                break

            if np.any(h_current >= 0.0):
                if self.time_step == 'm':
                    h_current[h_current >= 0.0] = -0.02
                    if h_current[0] == -0.02:
                        sat_val = self.soil_profile.th_s[0]
                        theta_current = self.prev_cond.th.copy()
                        addition = theta_current[0] + Infl / self.dz[0]
                        runoff = max(0, addition - sat_val) * self.dz[0]
                        Infl = max(0, Infl - runoff)
                        theta_current[0] = min(sat_val, addition)
                        theta_current += diff_water_content
                        epsilon = 1e-7
                        no_deep_perc = True
                        theta_current_clamped = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
                        h_current = psi_fun(theta_current_clamped, self.pars)
                        print('No Convergence')
                        converged = True
                        dry_soil = False
                        break
                else:
                    converged = False
                    break


            if iter > 18:
                if max_diff > 10.0 and max(abs(h_current)) > 100.0 and R == 0.0:
                    sat_val = self.soil_profile.th_s[0]
                    theta_current = self.prev_cond.th.copy()
                    addition = theta_current[0] + Infl / self.dz[0]
                    runoff = max(0, addition - sat_val) * self.dz[0]
                    Infl = max(0, Infl - runoff)
                    theta_current[0] = min(sat_val, addition)
                    theta_current += diff_water_content
                    epsilon = 1e-7
                    no_deep_perc = True
                    theta_current_clamped = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
                    h_current = psi_fun(theta_current_clamped, self.pars)
                    print('No Convergence')
                    converged = True
                    dry_soil = False
                    break

            if iter > 48 and max_diff < max(abs(h_current)) * 0.005:
                converged = True

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

            # if self.time_step == 'm' and max_diff > 20.0:
            #     print('No Convergence')
            #     converged = True

            if converged or divergence_count > 10:
                break

        # update theta/K/C

        if max_diff < 0.01:
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

            # elif self.time_step == 'm':
            #     solver = RichardEquationSolver(self.soil_profile, new_cond, time_step='s')
            if solver:
                dry_soil = False
                diff_water_sub = diff_water_content / solver.Nt
                sub_step_theta_prev = theta_prev_prev.copy()
                new_cond_sub = copy.deepcopy(new_cond)

                for _step in range(solver.Nt):
                    new_cond_sub.th = sub_step_theta_prev + diff_water_sub

                    converged, theta_current, _deep_perc, _runoff, _infl, K_current, C_current, h_current = solver.solve(
                        step * self.Nt + _step, new_cond_sub, irrigation / solver.Nt, rainfall / solver.Nt)

                    sub_step_theta_prev = theta_current.copy()

                    runoff += _runoff
                    Infl += _infl
                    deep_percolation += _deep_perc
        elif not converged:
            if dry_soil:
                sat_val = self.soil_profile.th_s[0]
                theta_current = self.prev_cond.th.copy()
                addition = theta_current[0] + Infl / self.dz[0]
                runoff = max(0, addition - sat_val) * self.dz[0]
                Infl = max(0, Infl - runoff)
                theta_current[0] = min(sat_val, addition)
                theta_current += diff_water_content
                epsilon = 1e-7
                no_deep_perc = True
                theta_current_clamped = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
                h_current = psi_fun(theta_current_clamped, self.pars)
                print('No Convergence')
                K_current = K(h_current, self.pars)
                q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)

            # else:
            #     # top soil layer is super wet
            #     sat_val = self.soil_profile.th_s[0]
            #     theta_current = self.prev_cond.th.copy()
            #     addition = theta_current[0] + Infl / self.dz[0]
            #     runoff = max(0, addition - sat_val) * self.dz[0]
            #     Infl = max(0, Infl - runoff)
            #     theta_current[0] = min(sat_val, addition)
            #     theta_current += diff_water_content
            #     epsilon = 1e-7
            #     no_deep_perc = True
            #     theta_current_clamped = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
            #     h_current = psi_fun(theta_current_clamped, self.pars)
            #     print('No Convergence')

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
            evap_traspir = sum(diff_water_content * self.dz) * 1000
            infilt = Infl * 1000
            err = prev_water - new_water + infilt - deep_perc + evap_traspir
            if abs(err) > 0.001:
                print('error')
        # if not converged and h_current[0] < -0.04:
        #     runoff = R

        self.prev_cond.th = theta_current

        new_theta = theta_current.flatten()
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
        for iter in range(self.max_iter):
            # at each step of picard iteration
            # values at iter = m
            # self.theta_val[:, step] = thetaf(h_current, self.pars)
            # self.K_val[:, step] = K(h_current, self.pars)
            # self.C_val[:, step] = C(h_current, self.pars)

            #K_step = self.K_val[:, step]
            # first layer is bit complicated as we need to set up boundary condition.
            K_half = mean(K_current[:-1], K_current[1:], self.dz.values)
            # K_half = 2 / (1/K_step[:-1] + 1/K_step[1:])

            # if there's rainfall or irrigation, updated h_current

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt, K_current[0],
                                   K_current[-1], diff_water_content, iter, R)



            flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0], self.Infiltration_So_Far, theta_current[0])
            b[0] += (flux / self.dz[0])  # downward positive flux
            Infl = flux * self.dt
            h_new = spsolve(csc_matrix(A), b)

            max_diff = np.max(np.abs(h_new - h_current))

            if max_diff > 2.0:
                converged = False
                break

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
        q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
        #self.deep_percolation_val[step] = deep_percolation

        new_theta = theta_current.flatten()#self.theta_val[:, step].flatten()
        return converged, new_theta, deep_percolation, runoff, Infl, K_current