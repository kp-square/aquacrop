import math

import numpy as np
from typing import Tuple, TYPE_CHECKING
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import copy
import itertools

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


# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/WR011i001p00102
def calculate_top_flux_infiltration(R, h_current, pars, dz, Ft, theta_val, theta_eff_sat):
    """
    Based on Green Ampt Equation modified for heterogeneous soil where infiltration can be limited by downward layers
    Args:
        R: rainfall
        h_current: matric potential of soil layers
        dz: depth of first layer
        pars: user defined parameters
    Returns:
        runoff: amount of water that runoffs the surface
        infl: flux that enters the soil surface for the timestep
    """
    available = R
    saturated_zone_ks = []
    for i in range(len(h_current)):
        vol_to_saturate = max(0, (theta_eff_sat[i] - theta_val[i]) * dz[i])
        saturated_zone_ks.append(pars['Ks'][i])
        if available > vol_to_saturate:
            available -= vol_to_saturate
        else:
            break
    if not saturated_zone_ks:
        Ks = pars['Ks'][0]
    else:
        Ks = min(saturated_zone_ks)

    del_theta = pars['thetaS'][0] - theta_val[0]
    h0 = h_current[0] # measured at center of top compartment
    # zc = dz/2.0 # 50cm
    if Ft == 0:
        Ft = 1e-6

    ft = Ks * (1 - h0 * del_theta / Ft) if Ft > 0 else Ks
    potential_infil_rate = max(0, ft)

    infl = min(R, potential_infil_rate)
    runoff = R - infl

    return infl, runoff


def topboundary_rainfall_evaporation(h_current, K_half, K_sat,
                                rainfall, evaporation, dz, hA, hS):
    # Calculate atmospheric equilibrium head
    h_surface = h_current[0]
    h_below = h_current[1]
    K_surface = K_half[0]
    # Net atmospheric demand
    net_demand = rainfall - evaporation
    boundry_type = 'flux'

    # Calculate soil hydraulic capacity
    dh_dx = (h_below - h_surface) / dz
    if net_demand > 0:  # infiltration
        q_soil_capacity = K_sat * ((hS - h_surface)/dz + 1)
    else:  # evaporation
        q_soil_capacity = K_surface * (dh_dx + 1)  # upward


    # Apply boundary condition logic
    if net_demand > 0:  # Rainfall dominant
        if h_surface <= hS:  # no ponding constraint violation
            q_actual = min(net_demand, q_soil_capacity)
            runoff = max(0.0, net_demand - q_soil_capacity)
        else:  # ponding would occur
            h_surface = hS
            q_actual = min(net_demand, q_soil_capacity)
            runoff = max(0.0, net_demand - q_soil_capacity)
            boundry_type='head'

    else:  # Evaporation dominant
        net_evap_demand = abs(net_demand)
        if h_surface >= hA:  # can sustain evaporation
            q_actual = -min(net_evap_demand, abs(q_soil_capacity))
        else:  # limited by atmospheric equilibrium
            h_surface = hA
            gradient_at_limit = (hA - h_below) / dz + 1.0
            q_flux_at_limit = K_surface * gradient_at_limit
            q_flux_at_limit = min(0.0, q_flux_at_limit)
            q_actual = -q_flux_at_limit
            boundry_type='head'
        runoff = 0

    return boundry_type, q_actual, runoff, h_surface


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


def assemble_system(K_half, C_val, theta_prev, theta_curr, h_curr, dz, dt, K_bottom, diff_water, z_nodes_center, has_water_table, water_table_depth):
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
    b[0] = (theta_prev[0] - theta_curr[0] + (C_val[0] * h_curr[0])) / dt - K_half[0] / dz[0] - diff_water[0]

    # middle compartments
    for i in range(1, n):
        dist_iminus1_i = (dz[i-1] + dz[i])/2.0
        A[i, i - 1] = - K_half[i - 1] / (dist_iminus1_i * dz[i])
        dist_iplus1_i = (dz[i] + dz[i+1]) / 2.0
        A[i, i] = C_val[i] / dt + K_half[i]/(dist_iplus1_i * dz[i]) + K_half[i - 1] / (dist_iminus1_i * dz[i])
        A[i, i + 1] = - K_half[i] / (dist_iplus1_i * dz[i])
        b[i] = (theta_prev[i] - theta_curr[i] + C_val[i] * h_curr[i]) / dt + (K_half[i-1] - K_half[i]) / dz[i]
        b[i] -= diff_water[i]

    # bottom boundary
    if has_water_table == 1:
        z_center_last_node = z_nodes_center[-1]
        dist_to_wt = water_table_depth - z_center_last_node
        head_last_node = h_curr[n] - z_center_last_node
        head_water_table = 0.0 - water_table_depth

        hydraulic_gradient = (head_water_table - head_last_node) / dist_to_wt
        q_bottom_boundary = -K_bottom * hydraulic_gradient  # K_bottom is K_current[-1]
        bottom_flux = q_bottom_boundary
    else:
        # no water table
        # free drainage
        bottom_flux = K_bottom

    if h_curr[n] > 0 and not has_water_table:
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
        b[n] = ((theta_prev[n] - theta_curr[n] + C_val[n] * h_curr[n]) / dt + (K_half[n-1] - bottom_flux)/last_dz)
        b[n] -= diff_water[-1]

    return A, b


def distribution_function(x, Lr, dz):
    Lr = max(Lr, 1e-5)
    cond1 = x < 0.2 * Lr
    cond2 = x < Lr  # This is implicitly (x >= 0.2 * Lr) & (x < Lr) in the nested where

    # Define the calculations for each condition
    val1 = 1.667 / Lr
    val2 = 2.0833 * (Lr - x) / (Lr ** 2)
    val3 = 0.0

    # Apply conditions using nested np.where for if/elif/else logic
    result = np.where(cond1, val1, np.where(cond2, val2, val3))

    # each compartment is a trapezoid with result representing mean of sides
    # dz is the height
    areas = result * dz
    normalized_areas = areas/max(np.sum(areas), 1e-5)
    return normalized_areas

def alpha_function(h, h1, h2, h3, h4):
    """
    Calculates the Feddes water stress response function, alpha(h).

    Args:
        h (float): Soil water pressure head [L].
        h1 (float): Anaerobiosis point (upper wet limit).
        h2 (float): Optimal moisture point (lower wet limit).
        h3 (float): Point where stress begins (upper dry limit).
        h4 (float): Wilting point (lower dry limit).

    Returns:
        float: The dimensionless stress factor alpha (0 to 1).
    """
    conditions = [
        (h > h2) & (h < h1),  # Linear ramp down (wet stress)
        (h >= h3) & (h <= h2),  # Optimal conditions
        (h > h4) & (h < h3),  # Linear ramp up (dry stress)
    ]

    # Define the corresponding values/calculations for each condition
    choices = [
        (h - h1) / (h2 - h1),
        1.0,
        (h - h4) / (h3 - h4),
    ]

    # np.select applies choices based on conditions. The default value of 0.0
    # handles the cases where h > h1 or h <= h4.
    return np.select(conditions, choices, default=0.0)

class RichardEquationSolver:
    def __init__(self, soil_profile, prev_cond, param_struct, time_step='d', use_root_uptake=True):
        self.pars = {}
        self.pars['thetaR'] = soil_profile.th_r
        self.pars['thetaS'] = soil_profile.th_s
        self.pars['alpha'] = soil_profile.alpha
        self.pars['n'] = soil_profile.n_param
        self.pars['m'] = 1 - 1 / self.pars['n']
        self.time_step = time_step.lower()
        self.soil_profile = soil_profile
        self.prev_cond = copy.deepcopy(prev_cond)
        self.param_struct = copy.deepcopy(param_struct)
        self.crop = self.param_struct.CropList[0]
        self.max_iter = 30
        self.ground_water_depth = prev_cond.z_gw
        self.has_water_table = param_struct.water_table
        self.use_root_uptake = use_root_uptake
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
        self.dt = 1
        self.L = sum(self.dz)
        self.psi = np.zeros((self.Nz, self.Nt))
        self.theta_val = np.zeros((self.Nz, self.Nt))
        self.runoff_val = np.zeros(self.Nt)
        self.deep_percolation_val = np.zeros(self.Nt)
        self.tolerance = 1e-5
        self.Infiltration_So_Far = 0.0
        self.z_nodes = np.array(list(itertools.accumulate(self.dz.values)))
        self.z_nodes_center = self.z_nodes - self.dz.values + self.dz.values/2.0
        self.SATURATION_THRESHOLD_h = -0.001
        self.h_eff_sat = np.full(self.Nz, self.SATURATION_THRESHOLD_h)
        self.theta_eff_sat = thetaf(self.h_eff_sat, self.pars)
        self.theta_eff_residual = thetaf(np.ones(self.Nz) * -300.0, self.pars)
        self.total_fallback_mins = 0
        self.T_surface_K = 273 + (prev_cond.temp_min + prev_cond.temp_max) / 2.0
        self.h_max = -0.01



    def compute_transpiration(self, h_current, Tr):
        b_x = distribution_function(self.z_nodes_center, self.prev_cond.z_root, self.dz.values)
        h1, h2, h4 = self.crop.h1, self.crop.h2, self.crop.h4
        if self.time_step == 'd':
            h3 = self.crop.h3_low if Tr < 5e-3 else self.crop.h3_high
        elif self.time_step == 'h':
            h3 = self.crop.h3_low if Tr < 0.5e-3 else self.crop.h3_high
        elif self.time_step == 'm':
            h3 = self.crop.h3_low if Tr < 0.5e-3/60 else self.crop.h3_high
        else:
            h3 = self.crop.h3_low if Tr < 0.5e-3 / 12 else self.crop.h3_high

        alpha_h = alpha_function(h_current, h1, h2, h3, h4)
        root_water_uptake = alpha_h * b_x * Tr
        return root_water_uptake/self.dz.values

    def solve(self, step, new_cond, irrigation, rainfall, EsPot, TrPot, rh = None):
        R = (rainfall + irrigation)/1000.0
        Es = EsPot/1000.0
        Tr = TrPot/1000.0
        if rh:
            # h_A = (RT/Mg)*ln(Hr): R,M and g are constants.
            h_A = 8.314 * self.T_surface_K / (0.018015*9.81) * math.log(rh/100)
        else:
            h_A = -500
        self.tolerance = 1e-5

        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)

        root_water_uptake = np.zeros(self.Nz)

        dry_soil = False
        no_deep_perc = False
        if any(np.array(self.prev_cond.th) < self.theta_eff_residual):
            dry_soil = True
            theta_prev = np.maximum(self.theta_eff_residual, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        converged = False
        prev_max_diff = 1.0
        infl = 0.0
        evaporation = 0.0
        transpiration = 0.0
        # Anderson Acceleration memory
        m_aa = 5
        h_hist = np.zeros((len(h_current), m_aa))
        f_hist = np.zeros((len(h_current), m_aa))
        beta = 1.0
        divergence_count = 0
        max_diff = 0.0
        for iter_count in range(self.max_iter):
            if R > 0.0 and self.time_step != 'm':
                converged = False
                break

            K_half = mean(K_current[:-1], K_current[1:], self.dz.values)

            boundary_type, q_flux, runoff, head_val = topboundary_rainfall_evaporation(h_current, K_half, self.pars['Ks'].iloc[0], R, Es, self.dz[0], h_A, self.h_max)
            #boundary_type, boundary_val, runoff, evaporation = calculate_surface_boundary(R, Es, h_A, h_current, K_half, self.pars['Ks'].iloc[0], self.dz[0], self.h_max)
            if boundary_type == 'head':
                h_current[0] = head_val
                if q_flux < 0.0:
                    evaporation = q_flux
            root_water_uptake = self.compute_transpiration(h_current, Tr)
            transpiration = sum(root_water_uptake * self.dz.values)
            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                   K_current[-1], root_water_uptake, self.z_nodes_center, self.has_water_table, self.ground_water_depth)

            if boundary_type == 'flux':
                b[0] += (q_flux / self.dz[0])  # downward positive flux
                if q_flux < 0.0:
                    evaporation = -q_flux
                else:
                    infl = q_flux

            try:
                b_mpi = b - (A @ h_current)
                delta_h = spsolve(csc_matrix(A), b_mpi)
            except:
                converged = False
                break

            max_diff = np.linalg.norm(delta_h)

            if max_diff < self.tolerance:
                converged = True

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

            k_aa = iter_count % m_aa
            h_hist[:, k_aa] = h_current
            f_hist[:, k_aa] = delta_h
            m_eff = min(iter_count+1, m_aa)
            if iter_count == 0:
                h_new = h_current + beta * delta_h
            else:
                f_diff = f_hist[:, 1:m_eff] - f_hist[:, 0:m_eff-1]
                gamma, _, _, _ = np.linalg.lstsq(f_diff, -delta_h, rcond=None)
                h_update_hist = h_hist[:, 1:m_eff] - h_hist[:, 0:m_eff-1]
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

        if iter_count == (self.max_iter - 1) and max_diff < 0.001:
            converged = True

        deep_percolation = 0.0
        fallback_used = False
        if not converged and self.time_step != 'm':
            fallback_used = True
            solver = None
            runoff = 0.0
            infl = 0.0
            evaporation = 0.0
            transpiration = 0.0
            if self.time_step == 'h':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, self.param_struct, 'sm', self.use_root_uptake)
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if self.time_step == 'sm':
                solver = RichardEquationSolver(self.soil_profile, self.prev_cond, self.param_struct, 'm', self.use_root_uptake)
                solver.Infiltration_So_Far = self.Infiltration_So_Far

            if solver:
                dry_soil = False
                sub_step_theta_prev = theta_prev_prev.copy()
                new_cond_sub = copy.deepcopy(new_cond)

                for _step in range(solver.Nt):
                    new_cond_sub.th = sub_step_theta_prev.copy()
                    converged, theta_current, _deep_perc, _runoff, _infl, _evap, _tr, K_current, h_current = solver.solve(
                        step * self.Nt + _step, new_cond_sub, irrigation / solver.Nt, rainfall / solver.Nt, EsPot/solver.Nt, TrPot/solver.Nt, rh)

                    sub_step_theta_prev = theta_current.copy()

                    runoff += _runoff
                    infl += _infl
                    deep_percolation += _deep_perc
                    evaporation += _evap
                    transpiration += _tr
                self.total_fallback_mins += solver.total_fallback_mins
        elif not converged:
            dry_soil = False
            theta_current, deep_percolation, runoff, infl, evaporation, transpiration, K_current, h_current = self.handle_non_convergence_bottom_up(R, theta_prev_prev, Es, Tr, h_A)

        self.Infiltration_So_Far += infl
        # calculate deep percolation
        if converged:
            if dry_soil:
                # dry layers
                mask = np.array(theta_prev_prev < self.theta_eff_residual)
                for i, val in enumerate(mask):
                    if val:
                        theta_current[i] = theta_prev_prev[i] + theta_current[i] - ref_prev_theta_current[i]
            if not fallback_used and not no_deep_perc:
                q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)
            prev_water = sum(self.prev_cond.th * self.dz) * 1000
            new_water = sum(theta_current * self.dz) * 1000
            deep_perc = deep_percolation * 1000
            transpir = transpiration * 1000
            evapo = evaporation * 1000
            infilt = infl * 1000
            err = prev_water - new_water + infilt - deep_perc - transpir - evapo
            if abs(err) > 0.02:
                print('error')

        self.prev_cond.th = theta_current
        new_theta = theta_current.flatten()
        return converged, new_theta, deep_percolation, runoff, infl, evaporation, transpiration, K_current, h_current


    def calculate_vapor_flux_upward(self, theta_new, h_current):
        """
        Calculates moisture change from upward vapor diffusion in the top 15 cm.

        This method implements the detailed step-by-step calculation for vapor
        flux between adjacent soil layers based on Fick's Law of diffusion.
        It is triggered when surface layers are dry.
        """
        if self.time_step == 'm':
            dt = 60
        elif self.time_step == 'sm':
            dt = 5 * 60
        elif self.time_step == 'h':
            dt = 24 * 60 * 60
        else:
            dt = 1.0
        # --- Physical Constants (SI Units) ---
        g = 9.81  # Acceleration due to gravity [m s⁻²]
        Mw = 0.018015  # Molar mass of water [kg mol⁻¹]
        R = 8.314  # Universal gas constant [J mol⁻¹ K⁻¹]
        Rv = R / Mw  # Specific gas constant for water vapor [J kg⁻¹ K⁻¹]
        RHO_WATER = 1000  # Density of liquid water [kg m⁻³]

        # --- Identify Target Layers (Top 15 cm) ---
        evap_zone_indices = np.where(self.z_nodes <= 0.15)[0]
        if len(evap_zone_indices) < 2:
            return theta_new  # Need at least two layers to calculate flux

        Nz_evap = len(evap_zone_indices)

        # --- Step 1: Characterize the Soil Layers ---
        theta_s_val = self.pars['thetaS']
        if not np.isscalar(theta_s_val):
            theta_s_val = theta_s_val[evap_zone_indices]
        theta_in_zone = theta_new[evap_zone_indices]

        # 1b. Calculate Air-Filled Porosity (beta)
        beta = np.maximum(0.0, theta_s_val - theta_in_zone)

        # --- Step 2: Determine the Vapor Density (rho_v) in Each Layer ---
        # 2a. Calculate Saturated Vapor Pressure (P_s) using Magnus-Tetens
        T_celsius = self.T_surface_K - 273.15
        P_s = 610.94 * np.exp((17.625 * T_celsius) / (T_celsius + 243.04))

        # 2b. Calculate Relative Humidity (hr) using the Kelvin equation
        h_in_zone = h_current[evap_zone_indices]
        hr = np.exp((h_in_zone * g * Mw) / (R * self.T_surface_K))

        # 2c. Calculate Vapor Density (rho_v) using the ideal gas law
        rho_v = (P_s * hr) / (Rv * self.T_surface_K)

        # --- Step 3: Determine the Effective Soil Vapor Diffusivity (D_soil) ---
        # 3a. Calculate Vapor Diffusivity in Air (D_vap)
        D_vap = (2.29e-5) * (self.T_surface_K / 273.15) ** 1.75

        # 3b. Calculate Effective Diffusivity in Soil (D_soil)
        D_soil = D_vap * (beta ** (2 / 3))

        # --- Step 4 & 5: Calculate Flux and Update Water Content ---
        # Iterate through interfaces from the bottom of the evap zone upwards
        for i in range(Nz_evap - 1, 0, -1):
            # i = lower layer index, i-1 = upper layer index (within evap_zone)
            abs_idx_i = evap_zone_indices[i]
            abs_idx_i_minus_1 = evap_zone_indices[i - 1]

            # 3c. Calculate Average Diffusivity Between Layers
            D_soil_avg = (D_soil[i] + D_soil[i - 1]) / 2.0

            # Calculate distance between layer centers
            dz_interface = self.z_nodes[abs_idx_i] - self.z_nodes[abs_idx_i_minus_1]

            # Calculate Upward Vapor Flux (q_v) using Fick's Law
            rho_gradient = (rho_v[i - 1] - rho_v[i]) / dz_interface
            q_v = -D_soil_avg * rho_gradient  # [kg m⁻² s⁻¹]

            if q_v <= 0: continue  # Only consider upward flux

            # Convert flux rate to a total volume of water for the timestep
            flux_mass = q_v * dt
            flux_volume = flux_mass / RHO_WATER  # [m] or [m³ m⁻²]

            # Limit flux by available water in the lower layer
            available_water = max(0, (theta_new[abs_idx_i] - self.theta_eff_residual[abs_idx_i])) * self.dz[abs_idx_i]
            flux_volume = min(flux_volume, available_water)

            # Limit flux by available space in the upper layer
            available_space = max(0, (self.theta_eff_sat[abs_idx_i_minus_1] - theta_new[abs_idx_i_minus_1])) * self.dz[
                abs_idx_i_minus_1]
            flux_volume = min(flux_volume, available_space)

            # Update water content in the two layers
            theta_new[abs_idx_i] -= flux_volume / self.dz[abs_idx_i]
            theta_new[abs_idx_i_minus_1] += flux_volume / self.dz[abs_idx_i_minus_1]

        return theta_new

    def handle_non_convergence_bottom_up(self, R, theta_prev, Es, Tr, h_A):
        """
        Handles non-convergence by using a robust, explicit finite-difference
        scheme. This method calculates water flux based on hydraulic head gradients.

        NOTE: This explicit scheme is subject to strict stability criteria and runs
        only with timestep of 1 minute.
        """
        # Start with the water content from the beginning of the timestep
        self.total_fallback_mins += 1
        steps = 1
        pars_sec = copy.deepcopy(self.pars)
        pars_sec['Ks'] = self.pars['Ks']
        theta_new = theta_prev.copy()

        clamped_theta = np.maximum(theta_new, self.theta_eff_residual)
        h_current = psi_fun(clamped_theta, pars_sec)

        K_temp = K(h_current, pars_sec)
        K_half = mean(K_temp[:-1], K_temp[1:], self.dz.values)
        root_uptake = self.compute_transpiration(h_current, Tr)
        transpiration = sum(root_uptake * self.dz)
        infl = 0.0
        evaporation = 0.0
        theta_new = theta_new - root_uptake

        # 4. Handle Deep Percolation (Bottom Boundary Condition)
        # Assume a "free drainage" condition where the gradient is 1 (gravity driven).
        # The flux is then equal to the hydraulic conductivity of the last layer.
        K_bottom = K_temp[-1]

        deep_percolation_volume = K_bottom * self.dt

        # Limit percolation by the amount of drainable water in the last layer.
        available_perc_volume = max(0, (theta_new[-1] - self.theta_eff_residual[-1])) * self.dz.iloc[-1]
        deep_percolation_volume = min(deep_percolation_volume, available_perc_volume)

        # Update the water content of the last layer.
        theta_new[-1] -= deep_percolation_volume / self.dz.iloc[-1]

        # 3. Handle Water Redistribution Between Layers
        # Calculate pressure head (psi) and hydraulic conductivity (K) from water content.
        # h_temp = psi_fun(theta_new, self.pars)

        # Iterate through all interfaces between the soil layers.
        for i in range(self.Nz - 1, 0, -1):
            # Approximate hydraulic conductivity at the interface between two layers.
            K_interface = K_half[i-1]

            # Calculate total hydraulic head (H = psi - z) for each layer.
            # Assumes self.z_nodes stores the depth (positive downwards) of the layer center.
            head_i = h_current[i] - self.z_nodes[i]
            head_i_minus_1 = h_current[i - 1] - self.z_nodes[i - 1]

            # Calculate the hydraulic gradient (dH/dz) between the two layers.
            dz_interface = (self.dz[i] + self.dz[i - 1]) / 2.0
            hydraulic_gradient = (head_i_minus_1 - head_i) / dz_interface

            # Calculate water flux per unit area using Darcy's Law (q = -K * dH/dz).
            flux_per_area = K_interface * hydraulic_gradient

            # Calculate the total volume of water moving across the interface in this timestep.
            flux_volume = flux_per_area * self.dt

            # Ensure flux does not create non-physical water content values.
            # Limit water flowing out of the upper layer.
            available_water_vol = max(0, (theta_new[i-1] - self.theta_eff_residual[i-1])) * self.dz[i-1]
            flux_volume = min(flux_volume, available_water_vol) if flux_volume > 0 else flux_volume

            # Limit water flowing into the lower layer.
            available_space_vol = max(0, (self.theta_eff_sat[i] - theta_new[i])) * self.dz[i]
            flux_volume = min(flux_volume, available_space_vol) if flux_volume > 0 else flux_volume

            # Handle upward flux (negative volume) limitations similarly.
            if flux_volume < 0:
                # Limit water flowing out of the lower layer.
                available_water_vol_lower = max(0, (theta_new[i] - self.theta_eff_residual[i])) * self.dz[i]
                flux_volume = max(flux_volume, -available_water_vol_lower)
                # Limit water flowing into the upper layer.
                available_space_vol_upper = max(0, (self.theta_eff_sat[i-1] - theta_new[i-1])) * self.dz[i-1]
                flux_volume = max(flux_volume, -available_space_vol_upper)

            # Update water content in the two layers based on the flux volume.
            # upper layer
            theta_new[i - 1] -= flux_volume / self.dz[i - 1]
            # lower layer
            theta_new[i] += flux_volume / self.dz[i]

        # 2. Handle Surface Boundary (Top Boundary Condition)
        boundary_type, q_flux, runoff, head_val = topboundary_rainfall_evaporation(h_current, K_half, self.pars['Ks'].iloc[0], R, Es,
                                                                              self.dz[0], h_A, self.h_max)

        if boundary_type == 'flux':
            theta_new[0] += q_flux / self.dz[0]
            if q_flux > 0.0:
                infl= q_flux
            else:
                evaporation = -q_flux
        else:
            h_current[0] = head_val

        # Final deep percolation is the volume per unit area.
        deep_percolation = deep_percolation_volume
        clamped_theta_new = np.maximum(theta_new, self.theta_eff_residual)
        h_new = psi_fun(clamped_theta_new, pars_sec)
        k_new = K(h_new, pars_sec)

        return theta_new, deep_percolation, runoff, infl, evaporation, transpiration, k_new, h_new

    def solve_daily(self, new_cond, irrigation, rainfall):
        R = (rainfall + irrigation) / 1000.0
        self.tolerance = 1e-5

        theta_prev = np.array(self.prev_cond.th)
        theta_prev_prev = theta_prev.copy()
        h_current = psi_fun(theta_prev, self.pars)

        if self.use_root_uptake:
            root_water_uptake = (new_cond.th - theta_prev_prev) / self.dt
        else:
            root_water_uptake = np.zeros(self.Nz)

        infl, runoff = calculate_top_flux_infiltration(R, h_current, self.pars, self.dz,
                                                       self.Infiltration_So_Far, theta_prev, self.theta_eff_sat)

        dry_soil = False
        if any(np.array(self.prev_cond.th) < self.theta_eff_residual):
            dry_soil = True
            theta_prev = np.maximum(self.theta_eff_residual, self.prev_cond.th)
            h_current = psi_fun(theta_prev, self.pars)

        theta_current = thetaf(h_current, self.pars)
        ref_prev_theta_current = theta_current.copy()
        K_current = K(h_current, self.pars)
        C_current = C(h_current, self.pars)
        converged = False
        prev_max_diff = 1.0

        # Anderson Acceleration memory
        m_aa = 5
        h_hist = np.zeros((len(h_current), m_aa))
        f_hist = np.zeros((len(h_current), m_aa))
        beta = 1.0
        divergence_count = 0
        max_diff = 0.0
        for iter_count in range(self.max_iter):

            K_half = mean(K_current[:-1], K_current[1:], self.dz.values)

            A, b = assemble_system(K_half, C_current, theta_prev, theta_current, h_current, self.dz, self.dt,
                                   K_current[-1], root_water_uptake, self.z_nodes_center, self.has_water_table,
                                   self.ground_water_depth)

            b[0] += (infl / self.dz[0])  # downward positive flux

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

            k_aa = iter_count % m_aa
            h_hist[:, k_aa] = h_current
            f_hist[:, k_aa] = delta_h
            m_eff = min(iter_count + 1, m_aa)
            if iter_count == 0:
                h_new = h_current + beta * delta_h
            else:
                f_diff = f_hist[:, 1:m_eff] - f_hist[:, 0:m_eff - 1]
                gamma, _, _, _ = np.linalg.lstsq(f_diff, -delta_h, rcond=None)
                h_update_hist = h_hist[:, 1:m_eff] - h_hist[:, 0:m_eff - 1]
                h_new = (h_current + delta_h) - np.dot(h_update_hist, gamma) - np.dot(f_diff, gamma)

                if beta != 1.0:
                    h_new = h_current + beta * (h_new - h_current)

            h_current = h_new

            if max_diff_differential > 5.0:
                break

            if np.any(h_current >= 0.0):
                break

            theta_current = thetaf(h_current, self.pars)
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

            if converged or divergence_count > 10:
                break

        if iter_count == (self.max_iter - 1) and max_diff < 0.001:
            converged = True

        deep_percolation = 0.0
        if converged:
            self.Infiltration_So_Far += infl
            q_bottom, deep_percolation = compute_deep_percolation(K_current, self.dt)

            if dry_soil:
                # dry layers
                mask = np.array(theta_prev_prev < self.theta_eff_residual)
                for i, val in enumerate(mask):
                    if val:
                        theta_current[i] = theta_prev_prev[i] + theta_current[i] - ref_prev_theta_current[i]


        self.prev_cond.th = theta_current

        new_theta = theta_current.flatten()
        return converged, new_theta, deep_percolation, runoff, infl, K_current, h_current