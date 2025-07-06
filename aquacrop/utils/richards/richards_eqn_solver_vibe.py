import math
import numpy as np
from typing import Tuple, TYPE_CHECKING
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve
import copy
import sys

if TYPE_CHECKING:
    from numpy import ndarray
    from aquacrop.entities.clockStruct import ClockStruct
    from aquacrop.entities.initParamVariables import InitialCondition
    from aquacrop.entities.paramStruct import ParamStruct
    from aquacrop.entities.output import Output
    from aquacrop.entities.crop import Crop


# --- Soil Hydraulic Property Functions (van Genuchten-Mualem) ---

def thetaf(psi, pars):
    """Calculate volumetric water content using the van Genuchten model."""
    psi_abs = np.abs(psi)
    Se = (1 + (psi_abs * pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    return pars['thetaR'] + (pars['thetaS'] - pars['thetaR']) * Se


def psi_fun(theta, pars):
    """Compute matric potential from the volumetric water content."""
    pw1 = 1 / pars['m']
    denom = np.maximum(theta - pars['thetaR'], 1e-9)
    Se_inv = (pars['thetaS'] - pars['thetaR']) / denom
    temp1 = np.maximum(Se_inv ** pw1 - 1, 0)
    return -(1.0 / pars['alpha']) * (temp1 ** (1 / pars['n']))


def C(psi, pars):
    """Calculate specific moisture capacity C(h) = d(theta)/dh."""
    psi_abs = np.abs(psi)
    nume = pars['m'] * pars['n'] * pars['alpha'] * (pars['alpha'] * psi_abs) ** (pars['n'] - 1)
    denom = (1 + (pars['alpha'] * psi_abs) ** pars['n']) ** (pars['m'] + 1)
    C_val = (pars['thetaS'] - pars['thetaR']) * nume / denom
    return np.maximum(C_val, 1e-9)


def K(psi, pars):
    """Compute hydraulic conductivity via the Mualem model."""
    psi_abs = np.abs(psi)
    Se = (1 + (psi_abs * pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    Se = np.minimum(Se, 1.0)
    k_val = pars['Ks'] * Se ** pars['neta'] * (1 - (1 - Se ** (1 / pars['m'])) ** pars['m']) ** 2
    return np.maximum(k_val, 1e-30)


# --- Flux and Boundary Condition Functions ---

def wieghted_geometric_mean(K_i, K_j, dz):
    """Calculate the weighted geometric mean for internodal hydraulic conductivity."""
    dist_i = dz[:-1] / 2.0
    dist_j = dz[1:] / 2.0
    total_dist = dist_i + dist_j
    log_K_i = np.log(K_i)
    log_K_j = np.log(K_j)
    weighted_log_mean = (log_K_i * dist_j + log_K_j * dist_i) / total_dist
    return np.exp(weighted_log_mean)


def calculate_top_flux_infiltration(R_rate, h_top, pars_top, Ft):
    """Calculate top flux using Green-Ampt, returning infiltration rate and runoff rate."""
    Ks = pars_top['Ks']
    theta_top = thetaf(np.array([h_top]), pars_top)[0]

    del_theta = pars_top['thetaS'] - theta_top
    Ft = max(Ft, 1e-6)

    ft = Ks * (1 - h_top * del_theta / Ft) if Ft > 0 else Ks
    infiltration_capacity = max(0, ft)

    infiltration_rate = min(R_rate, infiltration_capacity)
    runoff_rate = R_rate - infiltration_rate
    return infiltration_rate, runoff_rate


# --- System Assembly (Vectorized and Robust) ---

def assemble_system_vectorized(K_half, C_val, dz, dt):
    """Vectorized assembly of the Modified Picard system matrix A."""
    n = len(C_val)
    dz_vals = dz.values

    if n <= 1:
        A = csc_matrix((n, n))
        if n == 1: A[0, 0] = C_val[0] / dt
        return A

    dist = (dz_vals[:-1] + dz_vals[1:]) / 2.0
    K_out = K_half / (dist * dz_vals[:-1])
    K_in = K_half / (dist * dz_vals[1:])

    main_diag = C_val / dt
    main_diag[:-1] += K_out
    main_diag[1:] += K_in

    upper_diag = np.append(-K_out, 0)
    lower_diag = np.insert(-K_in, 0, 0)

    diagonals_data = np.array([main_diag, upper_diag, lower_diag])
    offsets = [0, 1, -1]

    A = spdiags(diagonals_data, offsets, n, n, format='csc')
    return A


class RichardEquationSolver:
    def __init__(self, soil_profile, prev_cond, time_step='d'):
        self.pars = {}
        self.pars['thetaR'] = soil_profile.th_r.values
        self.pars['thetaS'] = soil_profile.th_s.values
        self.pars['alpha'] = soil_profile.alpha.values
        self.pars['n'] = soil_profile.n_param.values
        self.pars['m'] = 1 - 1 / self.pars['n']

        self.time_step = time_step.lower()
        self.soil_profile = soil_profile
        self.prev_cond = copy.deepcopy(prev_cond)

        time_conversions = {'s': 24 * 60 * 60, 'm': 24 * 60, 'h': 24, 'd': 1}
        self.time_factor = time_conversions[self.time_step]

        # NEW: Add a sanity check for Ksat input values
        if np.any(soil_profile.Ksat.values > 20000):
            print(
                f"DIAGNOSTIC WARNING: Extremely high Ksat value detected in soil profile (> 20,000 mm/day). Please check soil file.",
                file=sys.stderr)
            print(f"Ksat Values (mm/day): {soil_profile.Ksat.values}", file=sys.stderr)

        self.pars['Ks'] = (soil_profile.Ksat.values / 1000) / self.time_factor
        self.Nt = {'s': 60, 'm': 60, 'h': 24, 'd': 1}[self.time_step]
        self.max_iter = 20
        self.tolerance = 1e-5

        self.pars['neta'] = soil_profile.neta.values
        self.dz = soil_profile.dz
        self.Nz = len(self.dz)
        self.dt = 1

        self.Infiltration_So_Far = 0.0
        self.theta_fc = thetaf(np.full(self.Nz, -1.0), self.pars)

    def _handle_non_convergence(self, theta_prev, R_rate, S_sink):
        """Last-resort fallback routine using a mass-conserving cascading bucket model."""
        print("WARNING: Solver failed to converge. Activating robust fallback routine.", file=sys.stderr)

        theta_new = theta_prev.copy()
        theta_new -= S_sink * self.dt

        h_prev = psi_fun(theta_prev, self.pars)
        pars_top = {key: val[0] for key, val in self.pars.items()}
        infiltration_rate, runoff_rate = calculate_top_flux_infiltration(R_rate, h_prev[0], pars_top,
                                                                         self.Infiltration_So_Far)

        infiltration_vol = infiltration_rate * self.dt
        if self.Nz > 0:
            theta_new[0] += infiltration_vol / self.dz[0]

        deep_percolation_vol = 0.0
        for i in range(self.Nz):
            excess_theta = max(0, theta_new[i] - self.theta_fc[i])
            if theta_new[i] > self.pars['thetaS'][i]:
                excess_theta = max(excess_theta, theta_new[i] - self.pars['thetaS'][i])

            theta_new[i] -= excess_theta
            water_to_drain = excess_theta * self.dz[i]

            if i < self.Nz - 1:
                theta_new[i + 1] += water_to_drain / self.dz[i + 1]
            else:
                deep_percolation_vol = water_to_drain

        theta_final = np.clip(theta_new, self.pars['thetaR'], self.pars['thetaS'])
        runoff_vol = runoff_rate * self.dt
        return theta_final, deep_percolation_vol, runoff_vol, infiltration_vol

    def solve(self, new_cond, irrigation, rainfall):
        R_rate = (rainfall + irrigation) / 1000.0 / self.dt

        theta_prev = np.array(self.prev_cond.th)
        h_current = psi_fun(theta_prev, self.pars)

        diff_water_content = new_cond.th - theta_prev
        S_sink = diff_water_content / self.dt

        converged = False
        for iter_num in range(self.max_iter):
            K_current = K(h_current, self.pars)
            C_current = C(h_current, self.pars)

            if self.Nz > 1:
                K_half = wieghted_geometric_mean(K_current[:-1], K_current[1:], self.dz.values)
            else:
                K_half = np.array([])

            A = assemble_system_vectorized(K_half, C_current, self.dz, self.dt)

            theta_curr_iter = thetaf(h_current, self.pars)
            residual = (theta_curr_iter - theta_prev) / self.dt - S_sink
            b = -residual

            try:
                delta_h = spsolve(csc_matrix(A), b)
                h_new = h_current + delta_h
            except Exception as e:
                print(f"Linear solver failed on iteration {iter_num}: {e}", file=sys.stderr)
                converged = False
                break

            if np.max(np.abs(delta_h)) < self.tolerance:
                converged = True
                h_current = h_new
                break

            h_current = h_new

        infiltration_vol, runoff_vol, deep_percolation = 0, 0, 0
        if not converged:
            if self.time_step == 'h':
                # Sub-stepping for hourly solver failure
                sub_solver = RichardEquationSolver(self.soil_profile, self.prev_cond, time_step='m')
                sub_solver.Infiltration_So_Far = self.Infiltration_So_Far
                theta_final = theta_prev
                for sub_step in range(sub_solver.Nt):
                    sub_new_th = theta_final + (diff_water_content / sub_solver.Nt)
                    sub_new_cond = type('obj', (object,), {'th': sub_new_th})()
                    _, th_out, dp_out, ro_out, inf_out, _, _, h_out = sub_solver.solve(sub_new_cond, 0,
                                                                                       R_rate * self.dt * 1000 / sub_solver.Nt)
                    theta_final = th_out
                    deep_percolation += dp_out / 1000
                    runoff_vol += ro_out / 1000
                    infiltration_vol += inf_out / 1000
                h_current = psi_fun(theta_final, self.pars)
            else:
                # Fallback for daily or minute solver failure
                theta_final, deep_percolation, runoff_vol, infiltration_vol = self._handle_non_convergence(theta_prev,
                                                                                                           R_rate,
                                                                                                           S_sink)
                h_current = psi_fun(theta_final, self.pars)
        else:
            # Path for successful convergence
            theta_final = thetaf(h_current, self.pars)
            K_final = K(h_current, self.pars)
            if self.Nz > 0:
                pars_top = {key: val[0] for key, val in self.pars.items()}
                infiltration_rate, runoff_rate = calculate_top_flux_infiltration(R_rate, h_current[0], pars_top,
                                                                                 self.Infiltration_So_Far)
                infiltration_vol = infiltration_rate * self.dt
                runoff_vol = runoff_rate * self.dt
                deep_percolation = K_final[-1] * self.dt
            else:
                infiltration_vol, deep_percolation = 0, 0
                runoff_vol = R_rate * self.dt

        self.Infiltration_So_Far += infiltration_vol
        self.prev_cond.th = theta_final
        K_final = K(h_current, self.pars)
        C_final = C(h_current, self.pars)

        # --- Final Mass Balance Check with Diagnostics ---
        et_vol = np.sum(S_sink * self.dt * self.dz.values)
        storage_change = np.sum((theta_final - theta_prev) * self.dz.values)
        error_m = (infiltration_vol - deep_percolation - et_vol) - storage_change
        error_mm = error_m * 1000

        # NEW: Print a detailed breakdown if the error is large
        if abs(error_mm) > 2.0:
            print("\nDIAGNOSTIC: Large mass balance error detected. Component values (mm):", file=sys.stderr)
            print(f"  - Infiltration (+):      {infiltration_vol * 1000:12.4f}", file=sys.stderr)
            print(f"  - Deep Percolation (-):  {deep_percolation * 1000:12.4f}", file=sys.stderr)
            print(f"  - ET Volume (-):         {et_vol * 1000:12.4f}", file=sys.stderr)
            print(f"  - Net Flux In-Out:       {(infiltration_vol - deep_percolation - et_vol) * 1000:12.4f}",
                  file=sys.stderr)
            print(f"  - Storage Change (dS):   {storage_change * 1000:12.4f}", file=sys.stderr)
            print(f"  - FINAL ERROR (Flux-dS): {error_mm:12.4f}\n", file=sys.stderr)

        return (True, theta_final.flatten(), deep_percolation, runoff_vol, infiltration_vol,
                K_final, C_final, h_current)