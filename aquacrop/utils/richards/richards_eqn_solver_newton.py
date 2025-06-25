import copy
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


def thetaf(psi, pars):
    '''Calculate volumetric water content using the van Genuchten model'''
    Se = (1 + (psi * -pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    return np.array(pars['thetaR'] + (pars['thetaS'] - pars['thetaR']) * Se)


def psi_fun(theta, pars):
    '''Compute matric potential from the volumetric water content'''
    epsilon = 1e-9
    pw1 = 1 / pars['m']
    theta_clamped = np.maximum(theta, pars['thetaR']+epsilon)
    denom = theta_clamped - pars['thetaR']
    denom[denom < 0.0] = 0.001
    Se = (pars['thetaS'] - pars['thetaR']) / denom
    temp1 = Se ** pw1 - 1
    return np.array(-(1.0 / pars['alpha']) * (temp1 ** (1 / pars['n'])))


def C(psi, pars):
    "Calculate specific moisture capacity C(h) = d(theta)/dh"
    nume = pars['m'] * pars['n'] * pars['alpha'] * (pars['alpha'] * np.abs(psi)) ** (pars['n'] - 1)
    denom = (1 + (pars['alpha'] * np.abs(psi)) ** (pars['n'])) ** (pars['m'] + 1)
    C_val = np.array(-(pars['thetaS'] - pars['thetaR']) * nume / denom * psi / (np.abs(psi) + 1e-10))
    return np.maximum(C_val, 1e-8)


def K(psi, pars):
    '''Compute hydraulic conductivity via the Mualem model'''
    Se = (1 + (psi * -pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    return np.array(pars['Ks'] * Se ** pars['neta'] * (1 - (1 - Se ** (1 / pars['m'])) ** pars['m']) ** 2)


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
    h0 = h_current[0]  # measured at center of top compartment
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
    """
    # Compute Darcy flux at bottom boundary (positive downward)
    q_bottom = K_current[-1]  # (m/hr)

    # Convert flux to volume per unit area over Δt (e.g., meters of water)
    percolation = q_bottom * dt  # dt

    return q_bottom, percolation  # (m)


# mean is used to calculate the conductivity on the surface
def mean(K_i, K_j, dz):
    weighted_mean = (K_i * (dz[1:] / 2) + K_j * dz[:-1] / 2) / ((dz[1:] + dz[:-1]) * 0.5)
    return weighted_mean


def assemble_newton_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_bottom, diff_water, iter=0, R=0):
    """
    Assembles the Jacobian matrix (J) and residual vector (F) for Newton's method.
    J is the system matrix (equivalent to 'A' in the original Picard implementation).
    F is the residual vector F(h) = J*h - b, where 'b' is the RHS of the Picard system.

    Outputs:
    J: Jacobian matrix for the system of equations.
    F: Residual vector F(h).
    """
    n = len(h_curr) - 1
    J = np.zeros((n + 1, n + 1))
    b_picard = np.zeros(n + 1)

    # Assemble Jacobian (J) and Picard RHS (b_picard)
    # Top boundary
    dist_iplus1_i = (dz[0] + dz[1]) / 2.0
    J[0, 0] = C_val[0] / dt + K_half[0] / (dist_iplus1_i * dz[0])
    J[0, 1] = -K_half[0] / (dist_iplus1_i * dz[0])
    b_picard[0] = (theta_prev[0] - theta_curr[0] + C_val[0] * h_curr[0]) / dt - K_half[0] / dz[0] + diff_water[0]

    # Middle compartments
    for i in range(1, n):
        dist_iminus1_i = (dz[i - 1] + dz[i]) / 2.0
        J[i, i - 1] = -K_half[i - 1] / (dist_iminus1_i * dz[i])
        dist_iplus1_i = (dz[i] + dz[i + 1]) / 2.0
        J[i, i] = C_val[i] / dt + K_half[i] / (dist_iplus1_i * dz[i]) + K_half[i - 1] / (dist_iminus1_i * dz[i])
        J[i, i + 1] = -K_half[i] / (dist_iplus1_i * dz[i])
        b_picard[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i - 1] - K_half[i]) / dz[i]
        if R == 0 or iter > 10:
            b_picard[i] += diff_water[i]

    # Bottom boundary
    if h_curr[n] > 0:
        J[n, n - 1] = 0.0
        J[n, n] = 1.0
        b_picard[n] = 0.0
    else:
        last_dz = dz.iloc[-1]
        dist_iminus1_i = (dz.iloc[-2] + last_dz) / 2.0
        J[n, n - 1] = -K_half[n - 1] / (last_dz * dist_iminus1_i)
        J[n, n] = C_val[n] / dt + K_half[n - 1] / (last_dz * dist_iminus1_i)
        # Free drainage
        b_picard[n] = (theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + (K_half[n - 1] - K_bottom) / last_dz
        if R == 0 or iter > 10:
            b_picard[n] += diff_water[-1]

    # Calculate the residual vector F(h) = J*h - b
    F = J @ h_curr - b_picard

    # Handle the special case for the saturated bottom boundary residual
    if h_curr[n] > 0:
        F[n] = h_curr[n]

    return J, F


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
        self.max_iter = 50

        if self.time_step == 's':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 60 * 60)  # m/sec
            self.Nt = 60

        elif self.time_step == 'm':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 60)  # m/minute
            self.Nt = 6

        elif self.time_step == 'sm':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / (24.0 * 10)  # m/(6 minutes)
            self.Nt = 10

        elif self.time_step == 'h':
            self.pars['Ks'] = (soil_profile.Ksat / 1000) / 24.0  # m/hr
            self.Nt = 24
        else:
            self.pars['Ks'] = (soil_profile.Ksat / 1000)  # m/day
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
        self.tolerance = 1e-8
        self.prev_cond = copy.deepcopy(prev_cond)
        self.Infiltration_So_Far = 0.0

    def solve(self, step, new_cond, irrigation, rainfall):
        # takes current hour as input
        # print(f'hour: {step}')
        R = (rainfall + irrigation) / 1000.0
        # If soil is super dry, i.e. in any layer moisture is below theta_r don't proceed
        self.tolerance = 1e-5
        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)
        dry_soil = False
        no_deep_perc = False
        if any(np.array(self.prev_cond.th) <= (self.soil_profile.th_r + 1e-4)):
            dry_soil = True
            epsilon = 1e-4
            theta_prev = np.maximum(self.pars['thetaR'] + epsilon, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)
            # self.prev_cond.th = new_cond.th

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        Infl = 0.0
        runoff = 0.0
        diff_water_content = (new_cond.th - theta_prev_prev) / self.dt
        converged = False

        for iter in range(self.max_iter):
            # Assemble Jacobian (J) and Residual (F)
            K_half = mean(K_current[:-1], K_current[1:], self.dz.values)
            J, F = assemble_newton_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                          K_current[-1], diff_water_content, iter, R)

            # Incorporate top boundary flux into the residual F[0]
            flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0],
                                                           self.Infiltration_So_Far, theta_current[0])
            F[0] -= (flux / self.dz[0])  # Downward positive flux
            Infl = flux * self.dt
            # Solve for the update step delta_h: J * delta_h = -F
            try:
                delta_h = spsolve(csc_matrix(J), -F)
            except Exception:
                converged = False
                break

            # Apply a damping factor for stability
            max_diff = np.max(np.abs(delta_h))
            damping_factor = 0.8 if max_diff > 2.0 else 1.0
            h_current += damping_factor * delta_h

            # Check for convergence based on the magnitude of the update
            if max_diff < self.tolerance:
                converged = True
                break

            if np.any(h_current >= 0.0):
                if self.time_step == 'm':
                    h_current[h_current >= 0.0] = 0.0
                    if h_current[0] == 0.0:
                        sat_val = self.soil_profile.th_s[0]
                        theta_current = self.prev_cond.th.copy()
                        addition = theta_current[0] + Infl/self.dz[0]
                        runoff = max(0, addition - sat_val) * self.dz[0]
                        Infl = max(0, Infl - runoff)
                        theta_current[0] = min(sat_val, addition)
                        theta_current += diff_water_content
                        epsilon = 1e-9
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

            if iter > 48:
                # too dry and non converging
                if max_diff > 10.0 and max(abs(h_current)) > 100.0 and R == 0.0:
                    sat_val = self.soil_profile.th_s[0]
                    epsilon = 1e-9
                    theta_current = self.prev_cond.th.copy()
                    addition = theta_current[0] + Infl / self.dz[0]
                    runoff = max(0, addition - sat_val) * self.dz[0]
                    Infl = max(0, Infl - runoff)
                    theta_current[0] = min(sat_val, addition)
                    theta_current += diff_water_content
                    clamped_theta = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
                    h_current = psi_fun(clamped_theta, self.pars)
                    no_deep_perc = True
                    converged = True
                    dry_soil = False
                    break

            if iter > 48 and max_diff < max(abs(h_current)) * 0.005:
                converged = True

            # Update state variables for the next iteration
            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

        # Post-convergence/iteration processing
        deep_percolation = 0.0
        fallback_used = False
        if not converged and self.time_step != 'm':
            fallback_used = True
            solver = None
            if self.time_step == 'h':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='sm')
                solver.Infiltration_So_Far = self.Infiltration_So_Far
            elif self.time_step == 'sm':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='m')
                solver.Infiltration_So_Far = self.Infiltration_So_Far
            if solver:
                dry_soil = False
                diff_water_sub = diff_water_content/solver.Nt
                new_cond_sub = copy.deepcopy(new_cond)
                new_cond_sub.th = self.prev_cond.th.copy()
                Infl = 0.0
                runoff = 0.0

                for _step in range(solver.Nt):
                    new_cond_sub.th += diff_water_sub
                    converged, theta_current, _deep_perc, _runoff, _infl, K_current, C_current, h_current = solver.solve(
                        _step, new_cond_sub, irrigation / solver.Nt, rainfall / solver.Nt)
                    new_cond_sub.th = theta_current
                    runoff += _runoff
                    Infl += _infl
                    deep_percolation += _deep_perc
        elif not converged:
            sat_val = self.soil_profile.th_s
            theta_current = self.prev_cond.th.copy()
            addition = theta_current[0] + Infl / self.dz[0]
            runoff = max(0, addition - sat_val[0]) * self.dz[0]
            Infl = max(0, Infl - runoff)
            theta_current[0] = min(sat_val[0], addition)
            theta_current += diff_water_content
            epsilon = 1e-9
            no_deep_perc = True
            theta_clamp = np.maximum(self.pars['thetaR'] + epsilon, theta_current)
            h_current = psi_fun(theta_clamp, self.pars)
            K_current = K(h_current, self.pars)
            q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)

        # else:
        #     flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0],
        #                                                    self.Infiltration_So_Far, theta_current[0])
        #     Infl = flux * self.dt

        self.Infiltration_So_Far += Infl

        if converged:
            if dry_soil:
                # dry layers
                mask = np.array(theta_prev_prev < (self.soil_profile.th_r + 1e-4))
                for i, val in enumerate(mask):
                    if val:
                        theta_current[i] = theta_prev_prev[i] + theta_current[i] - ref_prev_theta_current[i]
            if not fallback_used and not no_deep_perc:
                q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
            prev_water = sum(self.prev_cond.th * self.dz)*1000
            new_water = sum(theta_current * self.dz)*1000
            deep_perc = deep_percolation*1000
            evap_traspir = sum(diff_water_content * self.dz)*1000
            infilt = Infl*1000
            err = prev_water - new_water + infilt - deep_perc + evap_traspir
            if abs(err) > 0.01:
                print('error')
            self.prev_cond.th = theta_current
        if not converged and not dry_soil:
            runoff += R

        new_theta = theta_current.flatten()

        return converged, new_theta, deep_percolation, runoff, Infl, K_current, C_current, h_current

    def solve_daily(self, new_cond, irrigation, rainfall):
        R = (rainfall + irrigation) / 1000.0
        theta_prev = np.array(self.prev_cond.th)
        h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)

        diff_water_content = new_cond.th - theta_prev
        converged = False
        Infl = 0.0

        for iter in range(self.max_iter):
            # Assemble Jacobian (J) and Residual (F)
            K_half = mean(K_current[:-1], K_current[1:], self.dz.values)
            J, F = assemble_newton_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                          K_current[-1], diff_water_content, iter, R)

            # Incorporate top boundary flux into the residual F[0]
            flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0],
                                                           self.Infiltration_So_Far, theta_current[0])
            F[0] -= (flux / self.dz[0])  # Downward positive flux

            # Solve for the update step delta_h: J * delta_h = -F
            try:
                delta_h = spsolve(csc_matrix(J), -F)
            except Exception:
                converged = False
                break

            # Apply a damping factor for stability
            damping_factor = 0.8 if np.max(np.abs(delta_h)) > 2.0 else 1.0
            h_current += damping_factor * delta_h

            # Check for convergence based on the magnitude of the update
            if np.max(np.abs(delta_h)) < self.tolerance:
                converged = True
                break

            if np.any(h_current >= 0.0):
                converged = False
                break

            # Update state variables for the next iteration
            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

        # Post-convergence calculations
        flux, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz[0], self.Infiltration_So_Far,
                                                       theta_current[0])
        Infl = flux * self.dt

        q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
        new_theta = thetaf(h_current, self.pars).flatten()

        return converged, new_theta, deep_percolation, runoff, Infl, K_current