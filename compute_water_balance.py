import numpy as np

from simulation_script import run_simulation_and_get_balance, str_to_bool
import types
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataset.dataobjects import SoilType, ExpData
import csv
from typing import List, Dict, Any
import argparse

def compute_balance_process( hourly, use_richards, use_irrigation, texture=None):
    # no. of cores  = 54
    df = pd.read_csv('crop_metadata.csv')
    df_filter = df[(df['crop_type'] != 'peanut')]
    # df_filter = df[(df['crop_type'] == 'cotton') & (df['sirp_id'] == 217) & (df['treatment_id'] == 3)]
    if texture:
        df_filter = df[(df['crop_type'] == 'corn') & (df['sirp_id'] == 314) & (df['treatment_id'] == 2)]

    wp = {'corn': 33.7, 'cotton':12.9}
    hi0 = {'corn': 0.58, 'cotton':0.33}
    allargs = []
    for _, row in df_filter.iterrows():
        crop_type = row.get('crop_type')
        args = {}
        args['crop_type'] = crop_type
        args['year'] = row.get('year')
        args['treatment_id'] = row.get('treatment_id')
        args['sirp_id'] = row.get('sirp_id')
        args['hourly'] = hourly
        args['use_richards'] = use_richards
        args['WP'] = wp[crop_type]
        args['HI0'] = hi0[crop_type]
        args['run_count'] = 1
        args['use_irrigation'] = use_irrigation
        args['texture'] = texture
        allargs.append(types.SimpleNamespace(**args))
    all_results = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(run_simulation_and_get_balance, allargs):
            all_results.append(result)

    postfix = '' if use_irrigation else '_no_irr'
    postfix += '' if not texture else f'_{texture}'
    if hourly and use_richards:
        filename = f'hourly_richards_balance{postfix}.csv'
    elif use_richards:
        filename = f'daily_richards_balance{postfix}.csv'
    else:
        filename = f'daily_aquacrop_balance{postfix}.csv'
    save_dicts_to_csv(all_results, filename)

def save_dicts_to_csv(data_list: List[Dict[str, Any]], file_path: str) -> None:
    """
    Saves a list of dictionaries to a CSV file.
    The keys of the first dictionary in the list are used as the CSV header.
    It is assumed that all dictionaries in the list share the same keys.
    """
    if not data_list:
        return

    with open(file_path, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=data_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(data_list)

def main():
    parser = argparse.ArgumentParser(description="Compute Water Balance")
    parser.add_argument("--hourly", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--use_richards", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--use_irrigation", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--texture", type=str, required=False, default=None)
    args = parser.parse_args()
    hourly = args.hourly
    use_richards = args.use_richards
    use_irrigation = args.use_irrigation
    texture = args.texture
    print(hourly, use_richards, texture)
    compute_balance_process(hourly, use_richards, use_irrigation, texture)

if __name__=='__main__':
    #compute_balance_process(False, False, False)
    main()

