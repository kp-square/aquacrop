import argparse
import types

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from dataset.dataobjects import SoilType, ExpData
import pickle
from datetime import datetime, timezone, timedelta
import time

'''
class ExpData:
    def __init__(self, treatment_id, sirp_id, crop_type, year, irr_method, fert_method, lint_yield, start_date, end_date, irr):
        self.crop_type = crop_type
        self.year = year
        self.irr_method = irr_method
        self.fert_method = fert_method
        self.lint_yield = lint_yield
        self.irr = irr # dataframe
        self.treatment_id = treatment_id
        self.start_date = start_date
        self.end_date = end_date
        self.sirp_id = sirp_id
        self.soil_types = None
'''


def run_simulation(args):
    with open('dataset/experimental_data.pkl', 'rb') as f:
        pickle_data = pickle.load(f)
    expobj = None
    for obj in pickle_data:
        if obj.crop_type.lower() == args.crop_type.lower() and int(obj.year) == args.year and int(obj.treatment_id) == args.treatment_id and int(obj.sirp_id) == args.sirp_id:
            expobj = obj
            break
    if expobj is not None:
        if args.hourly and args.use_richards:
            weather_file_path = get_filepath('georgia_climate_hourly.txt')
        else:
            weather_file_path = get_filepath('georgia_climate_daily.txt')

        start_date = datetime.fromtimestamp(expobj.start_date / 1e9, tz=timezone.utc)
        end_date = datetime.fromtimestamp(expobj.end_date / 1e9, tz=timezone.utc) + timedelta(days=30)

        irr_sch = expobj.irr[['DATE', 'irr_depth']]
        desc = irr_sch.isna().sum()
        irr_sch.rename({'DATE': 'Date', 'irr_depth': 'Depth'}, axis=1, inplace=True)
        irrmethod = IrrigationManagement(irrigation_method=3, Schedule=irr_sch)

        dzz = []
        soil_types = []
        prev = 0.0
        for typ in expobj.soil_types:
            typ.depth = round(typ.depth, 2)
            splits = typ.soil_type.split(' ')
            splits = [x.capitalize() for x in splits]
            typ.soil_type = ''.join(splits)
            typ.soil_type = 'SandyClayLoam' if typ.soil_type.lower() == 'SadnyClayLoam' else typ.soil_type
            dzz.append(round(typ.depth - prev, 2))
            soil_types.append(typ.soil_type)
            prev = typ.depth

        # Make at least 10 layers of soil, extend the last layer
        while len(dzz) < 10:
            dzz.append(dzz[-1])
            soil_types.append(soil_types[-1])

        soil = Soil(soil_type=soil_types, dz=dzz)
        step_size = 'H' if args.hourly else 'D'

        crop_type = 'Maize' if args.crop_type == 'corn' else args.crop_type
        model_os = AquaCropModel(
            sim_start_time=f'{start_date.year}/{start_date.month}/{start_date.day}',
            sim_end_time=f'{end_date.year}/{end_date.month}/{end_date.day}',
            weather_df=prepare_weather(weather_file_path, hourly = args.hourly),
            soil=soil,
            crop=Crop(crop_type, planting_date=f'{start_date.month}/{start_date.day}', WP=args.WP, HI0=args.HI0),
            initial_water_content=InitialWaterContent(value=['FC']*soil.nComp, method="Layer", depth_layer=soil.profile.Layer),
            irrigation_management=irrmethod,
            step_size=step_size,
            use_richards=args.use_richards
        )

        model_os.run_model(till_termination=True)
        model_results_df = model_os.get_simulation_results()
        simulated_yield = model_results_df['Dry yield (tonne/ha)'].iloc[0]
        # sq_err = (simulated_yield_obj.value - expobj.lint_yield)**2
        with open('results.txt', 'a') as file:
            file.write(f'{args.run_count}\t{args.crop_type}\t{simulated_yield}\t{expobj.lint_yield}')

def str_to_bool(value: str) -> bool:
    """Converts a string representation of truth to True or False."""
    if isinstance(value, bool): # If already a bool, return it
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid boolean value. Use 'true' or 'false'.")

def main():
    parser = argparse.ArgumentParser()
    # crop_type, year, treatment_id, sirp_id, hourly, use_richards, WP, HI0, run_count
    parser.add_argument("--crop_type", type=str, required=True, default='corn')
    parser.add_argument("--year", type=int, required=True, default=2018)
    parser.add_argument("--treatment_id", type=int, required=True, default=2)
    parser.add_argument("--sirp_id", type=int, required=True, default=118)
    parser.add_argument("--hourly", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--use_richards", type=str_to_bool, required=True, metavar='{true/false}', default=True)
    parser.add_argument("--WP", type=float, required=True, default=33.3)
    parser.add_argument("--HI0", type=float, required=True, default=0.48)
    parser.add_argument("--run_count", type=int, required=True, default=1)
    args = parser.parse_args()
    run_simulation(args)


if __name__=='__main__':
    x = 123
    args = {}
    args['crop_type'] = 'Cotton'
    args['year'] = 2019
    args['treatment_id'] = 3
    args['sirp_id'] = 114
    args['hourly'] = True
    args['use_richards'] = True
    args['WP'] = 12
    args['HI0'] = 0.27
    args['run_count'] = 1
    run_simulation(types.SimpleNamespace(**args))