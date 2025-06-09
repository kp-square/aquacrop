import numpy as np

from simulation_script import run_simulation
import types
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataset.dataobjects import SoilType, ExpData
import optuna


def hp_optimizer(crop_type, years, wp_range, hi0_range, hourly, use_richards):
    #optimize wp, hi0 with minimum average error obtained from run_processes
    n_trials = 10
    def objective(trial):
        wp = trial.suggest_float('wp', wp_range[0], wp_range[1])
        hi0 = trial.suggest_float('hi0', hi0_range[0], hi0_range[1])

        try:
            error = run_processes(crop_type, years, wp, hi0, trial.number, hourly, use_richards)
            return error
        except Exception as e:
            print(f'Error in trial {trial.number}: {e}')
            return float('inf')

    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials = n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_error = study.best_value

    return best_params['wp'], best_params['hi0'], best_error

def run_processes(crop_type, years, wp, hi0, run_count, hourly, use_richards):
    df = pd.read_csv('crop_metadata.csv')
    df_filter = df[(df['crop_type'] == crop_type) & (df['year'].isin(years))][:8]
    allargs = []
    for _, row in df_filter.iterrows():
        args = {}
        args['crop_type'] = row.get('crop_type')
        args['year'] = row.get('year')
        args['treatment_id'] = row.get('treatment_id')
        args['sirp_id'] = row.get('sirp_id')
        args['hourly'] = hourly
        args['use_richards'] = use_richards
        args['WP'] = wp
        args['HI0'] = hi0
        args['run_count'] = run_count
        allargs.append(types.SimpleNamespace(**args))
    sq_errors = []
    with ProcessPoolExecutor() as executor:
        for err in executor.map(run_simulation, allargs):
            sq_errors.append(err)
    return np.mean(sq_errors)


def main():
    crop_type = 'corn'
    all_years = [2018, 2019, 2020]
    wp_range = (20.0, 35.0)
    hi_range = (0.45, 0.65)
    trains = [(0,1)]#, (1,2), (0,2)]
    model_configs = {'Old Aquacrop':(True, True)} # 'Richards daily':(True, False), 'Richards Hourly':(True, True)}
    for model_name in model_configs.keys():
        hourly, use_richards = model_configs[model_name]
        for train in trains:
            train_years = [all_years[i] for i in train]
            test_years = filter(lambda x: x not in train_years, all_years)
            test_years = list(test_years)
            wp, hi, best_error = hp_optimizer(crop_type, train_years, wp_range, hi_range, hourly, use_richards)
            test_err = run_processes(crop_type, test_years, wp, hi, 1, hourly, use_richards)
            print(model_name, test_years, wp, hi, crop_type, test_err, best_error)

    # crop_type = 'cotton'
    # all_years = [2019, 2020, 2021]
    # wp_range = (10.0, 20.0)
    # hi_range = (0.20, 0.35)
    # hi_final_range = (0.30, 1.0)
    # trains = [(0, 1), (1, 2), (0, 2)]
    # for model_name in model_configs.keys():
    #     hourly, use_richards = model_configs[model_name]
    #     for train in trains:
    #         train_years = [all_years[i] for i in train]
    #         test_years = filter(lambda x: x not in train_years, all_years)
    #         wp, hi, hi_final = hp_optimizer(crop_type, test_years, wp_range, hi_range, hi_final_range, hourly, use_richards)
    #         test_err = run_processes(crop_type, test_years, wp, hi, hi_final, 1, hourly, use_richards)
    #         train_err = run_processes(crop_type, train_years, wp, hi, hi_final, 1, hourly, use_richards)
    #         print(model_name, test_years, wp, hi, hi_final, crop_type, test_err, train_err)


    # avg_err = run_processes('cotton', [2019, 2020], 12.5, 0.27, 0.75, 1)
    # print(avg_err)

if __name__=='__main__':
    main()

