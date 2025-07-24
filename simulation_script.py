import argparse
import types

from aquacrop import AquaCropModel, SoilGeorgia, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from dataset.dataobjects import SoilType, ExpData
import pickle
from datetime import datetime, timezone, timedelta
import time

'''
class ExpData:
    def __init__(self, treatment_id, sirp_id, crop_type, year, irr_method, fert_method, crop_yield, start_date, end_date, irr):
        self.crop_type = crop_type
        self.year = year
        self.irr_method = irr_method
        self.fert_method = fert_method
        self.crop_yield = crop_yield
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
            weather_file_path = get_filepath('georgia_climate_hourly_rh.txt')
        else:
            weather_file_path = get_filepath('georgia_climate_daily.txt')

        start_date = datetime.fromtimestamp(expobj.start_date / 1e9, tz=timezone.utc)
        end_date = datetime.fromtimestamp(expobj.end_date / 1e9, tz=timezone.utc) + timedelta(days=30)

        irr_sch = expobj.irr[['DATE', 'irr_depth']]
        desc = irr_sch.isna().sum()
        irr_sch = irr_sch.rename({'DATE': 'Date', 'irr_depth': 'Depth'}, axis=1)
        if args.use_irrigation:
            irrmethod = IrrigationManagement(irrigation_method=3, Schedule=irr_sch)
        else:
            irrmethod = IrrigationManagement(irrigation_method=0)

        soil_types, dzz = define_soil_texture(expobj, args.use_richards, args.texture)

        soil = SoilGeorgia(soil_type=soil_types, dz=dzz, evap_z_surf=0.06, evap_z_min=0.06, evap_z_max=0.10)
        step_size = 'H' if args.hourly else 'D'
        # source: https://open.clemson.edu/cgi/viewcontent.cgi?article=2297&context=all_theses
        cotton_params = {'CGC_CD':0.10, 'CDC_CD':0.029, 'CCx':0.98, 'Kcb':1.1, 'Zx':1.2, 'WP':args.WP, 'HI0':args.HI0, 'EmergenceCD':3, 'SenescenceCD': 100, 'MaturityCD': 160, 'FloweringCD':42, 'Tbase':15.6, "HIstartCD":64, 'YldFormCD':95, 'SwitchGDD':1,  'h1':-0.1, 'h2':-0.25, 'h3_low': -8.0, 'h3_high':-2.0, 'h4':-11.0}
        # I may need to test for corn later
        # source: https://extension.missouri.edu/media/wysiwyg/Extensiondata/CountyPages/Scott/Irrigation/Estimated-Water-Use-Corn-Georgia.pdf
        # source: https://www.sciencedirect.com/science/article/pii/S0378377418317128?casa_token=3li4XZy-0EIAAAAA:sz1Z1SeThgCzLiVUpZ6JqB0Le_b9ipfAZsexuLDjgHRmyLZi9jzQUP-HvFvriLw1TZioBxnjHA
        corn_params = {'CCx':0.94, 'Zx':2.1, 'CGC_CD':0.137, 'Kcb':1.05, 'HI0':args.HI0, 'WP':args.WP, 'EmergenceCD':7, 'MaxRootingCD':79, 'SenescenceCD':105, 'Tbase':10, 'MaturityCD':122, 'FloweringCD':15, "HIstartCD":80, 'YldFormCD':35, 'SwitchGDD':1}
        params = {'cotton': cotton_params, 'corn': corn_params}
        crop_type = 'Maize' if args.crop_type == 'corn' else args.crop_type.capitalize()
        model_os = AquaCropModel(
            sim_start_time=f'{start_date.year}/{start_date.month}/{start_date.day}',
            sim_end_time=f'{end_date.year}/{end_date.month}/{end_date.day}',
            weather_df=prepare_weather(weather_file_path, hourly = args.hourly),
            soil=soil,
            crop=Crop(crop_type, planting_date=f'{start_date.month}/{start_date.day}', **params[args.crop_type]),
            initial_water_content=InitialWaterContent(value=['FC']*soil.nComp, method="Layer", depth_layer=soil.profile.Layer),
            irrigation_management=irrmethod,
            step_size=step_size,
            use_richards=args.use_richards
        )

        model_os.run_model(till_termination=True)
        model_results_df = model_os.get_simulation_results()
        return model_results_df, expobj

def define_soil_texture(expobj, use_richards=False, texture=None):
    dz = []
    soil_types = []
    prev = 0.0

    for typ in expobj.soil_types:
        typ.depth = round(typ.depth, 2)
        prev_depth = prev  # Start from where previous layer ended
        levels = [0.1, 0.1, 0.2]
        if use_richards:
            levels = [0.02, 0.05, 0.10]
        while prev_depth < typ.depth:
            # Determine compartment size based on current cumulative depth
            if sum(dz) < 0.2:  # First 20 cm: levels[0] m compartments
                compartment_size = levels[0]
            elif sum(dz) < 1.0:  # Next 80 cm (up to 1 m): levels[1] m compartments
                compartment_size = levels[1]
            else:  # Rest: levels[2] m compartments
                compartment_size = levels[2]

            # Calculate next depth
            next_depth = prev_depth + compartment_size

            # Handle the last compartment if remaining depth is small
            if typ.depth - prev_depth < 1.5 * compartment_size:
                next_depth = typ.depth

            if not texture:
                if not typ.soil_type:
                    typ.soil_type = 'Loamy Fine Sand'
                splits = typ.soil_type.split(' ')
                splits = [x.capitalize() for x in splits]
                soil_type = ''.join(splits)
                soil_type = 'SandyClayLoam' if soil_type == 'SadnyClayLoam' else soil_type
                soil_types.append(soil_type)
            else:
                soil_types.append(texture)

            dz.append(round(next_depth - prev_depth, 2))
            prev_depth = next_depth

        prev = typ.depth

    # Make at least 40 layers of soil, extend the last layer
    while sum(dz) < 3.0:
        dz.append(0.20)
        soil_types.append(soil_types[-1])

    return soil_types, dz

def run_simulation_and_get_balance(args):
    t1 = time.time()
    model_results_df, expobj = run_simulation(args)
    result  = {}
    result['irr'] = model_results_df["Seasonal irrigation (mm)"].iloc[0]
    result['rain'] = model_results_df["Seasonal rainfall (mm)"].iloc[0]
    result['es'] = model_results_df["Evaporation (mm)"].iloc[0]
    result['tr'] = model_results_df["Transpiration (mm)"].iloc[0]
    result['dp'] = model_results_df["Deep Percolation (mm)"].iloc[0]
    result['runoff'] = model_results_df["Runoff (mm)"].iloc[0]
    result['infl'] = model_results_df["Seasonal Infiltration (mm)"].iloc[0]
    result['balance'] = model_results_df["Balance (mm)"].iloc[0]
    result['sim_yield'] = model_results_df["Dry yield (tonne/ha)"].iloc[0]
    result['actual_yield'] = expobj.crop_yield
    result['year'] = expobj.year
    result['crop'] = expobj.crop_type
    result['sirp_id'] = expobj.sirp_id
    result['treatment_id'] = expobj.treatment_id
    result['irr_method'] = expobj.fert_method
    result['richards_fallback'] = model_results_df["Richards Total Fallback"].iloc[0]
    result['texture'] = args.texture
    result['simulation_time'] = time.time() - t1
    for key in result.keys():
        if isinstance(result[key], float):
            result[key] = round(result[key], 2)
    return result

def run_simulation_and_get_yield_error(args):
    model_results_df, expobj = run_simulation(args)
    simulated_yield = model_results_df['Dry yield (tonne/ha)'].iloc[0]
    abs_err = abs(simulated_yield - expobj.crop_yield)
    sq_err = abs_err ** 2
    perc_err = abs_err/expobj.crop_yield
    return sq_err, perc_err

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
    parser.add_argument("--use_irrigation", type=str_to_bool, required=True, default=True)
    parser.add_argument("--texture", type=str, required=False, default=None)
    args = parser.parse_args()
    run_simulation(args)


if __name__=='__main__':
    x = 123
    args = {}
    args['crop_type'] = 'cotton'
    args['year'] = 2020
    args['treatment_id'] = 6
    args['sirp_id'] = 222
    args['hourly'] = True
    args['use_richards'] = True
    args['WP'] = 12
    args['HI0'] = 0.27
    args['run_count'] = 1
    args['use_irrigation'] = True
    args['texture'] = None
    run_simulation(types.SimpleNamespace(**args))