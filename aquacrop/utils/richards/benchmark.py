from richards_eqn_solver_newton import RichardEquationSolver, thetaf
from types import SimpleNamespace
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
def parse_hydrus_output(file_path):
    data = {}
    current_time = None
    with open(file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line.startswith('Time:'):
                current_time = float(stripped_line.split()[-1])
                data[current_time] = []
            elif current_time is not None and stripped_line and stripped_line.split()[0].isdigit():
                try:
                    parts = stripped_line.split()
                    depth = float(parts[1])
                    head = float(parts[2])
                    moisture = float(parts[3])
                    data[current_time].append({'Depth': depth, 'Head': head, 'Moisture': moisture})
                except (ValueError, IndexError):
                    continue
            elif stripped_line.lower() == 'end':
                current_time = None
    dataf = {}
    for cur_time in data.keys():
        hvalue = []
        thvalue = []
        for obj in data[cur_time]:
            hvalue.append(obj['Head'])
            thvalue.append(obj['Moisture'])
        dataf[cur_time] = {'head': hvalue, 'theta': thvalue}
    return dataf

def plot(hydrus, richards, depth):
    hydrus_head, hydrus_theta = hydrus
    richards_head, richards_theta = richards
    # implement code to plot head vs depth and theta vs depth for both in same graph, make one as (dashes, red) and the other as (line, green)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Head vs Depth
    ax1.plot(hydrus_head, depth, color='red', linestyle='--', label='Hydrus Head')
    ax1.plot(richards_head, depth, color='green', linestyle='-', label='Richards Head')
    ax1.set_xlabel("Head [L]")
    ax1.set_ylabel("Depth [L]")
    ax1.set_title("Head vs. Depth")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Plot Moisture vs Depth
    ax2.plot(hydrus_theta, depth, color='red', linestyle='--', label='Hydrus Moisture')
    ax2.plot(richards_theta, depth, color='green', linestyle='-', label='Richards Moisture')
    ax2.set_xlabel("Moisture [-]")
    ax2.set_ylabel("Depth [L]")
    ax2.set_title("Moisture vs. Depth")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('hydrus_vs_richards.png')
    plt.show()


def main():
    sim_time_hours = 24
    profile_depth = 1.0 #meters
    num_layers = 101
    initial_pressure_head = -1.329

    precip = [0.0]*5 + [0.002, 0.004, 0.006] + [0.0]*16
    evap = [0.0]*8 + [0.0005]*3 + [0.001]*3 + [0.0005]*3 + [0.0] * 7
    soil_params = {
        'th_r': 0.078,
        'th_s': 0.43,
        'alpha': 3.6,
        'n_param': 1.56,
        'Ksat': 249.6,
        'neta': 0.5
    }

    dz_array = np.ones(num_layers) * (profile_depth / num_layers)
    soil_profile = SimpleNamespace(
        th_r = np.full(num_layers, soil_params['th_r']),
        th_s = np.full(num_layers, soil_params['th_s']),
        alpha = np.full(num_layers, soil_params['alpha']),
        n_param = np.full(num_layers, soil_params['n_param']),
        Ksat = np.full(num_layers, soil_params['Ksat']),
        dz = pd.Series(dz_array),
        neta = np.full(num_layers, soil_params['neta'])
    )

    initial_h = np.full(num_layers, initial_pressure_head)
    pars = {}
    pars['thetaR'] = soil_profile.th_r
    pars['thetaS'] = soil_profile.th_s
    pars['alpha'] = soil_profile.alpha
    pars['n'] = soil_profile.n_param
    pars['m'] = 1 - 1/pars['n']
    pars['neta'] = 0.5

    initial_th = thetaf(initial_h, pars)
    prev_cond = SimpleNamespace(th=initial_th)
    new_cond = copy.deepcopy(prev_cond)

    solver = RichardEquationSolver(soil_profile, prev_cond, time_step='h')
    all_h_val = {}
    all_th_val = {}
    all_h_val[0] = initial_h
    all_th_val[0] = initial_th
    for i in range(24):
        new_cond.th[0] -= evap[i]/dz_array[0]
        converged, new_theta, deep_percolation, runoff, Infl, K_current, C_current, h_current = solver.solve(i, new_cond, 0.0, precip[i]*1000)
        all_h_val[i+1] = h_current
        all_th_val[i+1] = new_theta
        if converged:
            new_cond.th = new_theta
        else:
            print('Not Converged')

    hydrus_data = parse_hydrus_output('../../../dataset/Hydrus_Nod_Inf.out')
    hydrus = (hydrus_data[24.0]['head'], hydrus_data[24.0]['theta'])
    richards = (all_h_val[24], all_th_val[24])
    dz_cum = [0.0]
    for val in dz_array:
        dz_cum.append(dz_cum[-1]+val)
    dz_cum = np.array(dz_cum)
    dz_mid = (dz_cum[:-1] + dz_cum[1:])/2.0

    plot(hydrus, richards, dz_mid)
    print('Done.')




if __name__ == '__main__':
    main()