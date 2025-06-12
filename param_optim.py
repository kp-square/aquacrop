import numpy as np

from simulation_script import run_simulation_and_get_yield_error, str_to_bool
import types
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataset.dataobjects import SoilType, ExpData
import optuna
import argparse


def hp_optimizer(crop_type, years, wp_range, hi0_range, hourly, use_richards):
    #optimize wp, hi0 with minimum average error obtained from run_processes
    n_trials = 100
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
    # number of cores = 18
    df = pd.read_csv('crop_metadata.csv')
    df_filter = df[(df['crop_type'] == crop_type) & (df['year'].isin(years))]
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
        for err in executor.map(run_simulation_and_get_yield_error, allargs):
            sq_errors.append(err)
    return np.mean(sq_errors)


def main():
    parser = argparse.ArgumentParser('optimize parameters')
    parser.add_argument('--crop_type', type=str, required=True)
    parser.add_argument('--test_year', type=int, required=True)
    parser.add_argument("--hourly", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--use_richards", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    args = parser.parse_args()
    crop_type = args.crop_type
    test_year = args.test_year
    hourly = args.hourly
    use_richards = args.use_richards
    assert(crop_type in ['corn', 'cotton'])
    if crop_type == 'cotton':
        all_years = [2019, 2020, 2021]
        wp_range = (8.0, 20.0)
        hi0_range = (0.20, 0.35)
    else:
        all_years = [2018, 2019, 2020]
        wp_range = (20.0, 35.0)
        hi0_range = (0.45, 0.65)
    assert (test_year in all_years)
    train_years = [y for y in all_years if y != test_year]
    test_years = [test_year]
    wp, hi, best_error = hp_optimizer(crop_type, train_years, wp_range, hi0_range, hourly, use_richards)
    test_err = run_processes(crop_type, test_years, wp, hi, 1, hourly, use_richards)
    if hourly and use_richards: model_name = 'Hourly_Richards_Aquacrop'
    elif not hourly and use_richards: model_name = 'Daily_Richards_Aquacrop'
    else: model_name = 'Daily_Aquacrop'
    with open(f'{model_name}_{test_year}_{crop_type}.txt', 'w') as file:
        file.write(f'wp: {wp}, hi0: {hi}, test_err: {test_err}, train_error: {best_error}')


if __name__=='__main__':
    # run_processes('corn', [2018], 12, 0.47, 1, True, True)
    main()

