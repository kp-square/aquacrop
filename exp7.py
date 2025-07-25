from aquacrop import AquaCropModel, Soil, SoilGeorgia, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from dataset.dataobjects import SoilType, ExpData
import pickle
from datetime import datetime, timezone, timedelta


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

with open('dataset/experimental_data.pkl', 'rb') as f:
    pickle_data = pickle.load(f)

expobj = pickle_data[0]

weather_file_path = get_filepath('georgia_climate_daily.txt')
start_date = datetime.fromtimestamp(expobj.start_date / 1e9, tz=timezone.utc)
end_date = datetime.fromtimestamp(expobj.end_date / 1e9, tz=timezone.utc) + timedelta(days=30)

irr_sch = expobj.irr[['DATE', 'irr_depth']]
irr_sch.rename({'DATE':'Date', 'irr_depth':'Depth'}, axis=1, inplace=True)
irrmethod = IrrigationManagement(irrigation_method=3, Schedule=irr_sch)
#irrmethod.Schedule = irr_sch

TARGET_TOP_LAYER_THICKNESS = 0.02
TOLERANCE = 0.005
dzz = []
soil_types = []
prev = 0.0
for typ in expobj.soil_types:
    typ.depth = round(typ.depth, 2)
    typ.soil_type = ''.join(typ.soil_type.split(' '))
    dzz.append(round(typ.depth - prev, 2))
    soil_types.append(typ.soil_type)
    prev = typ.depth

# Ensure the top layer has thickness equal to 2 cm only.
if dzz and dzz[0] > (TARGET_TOP_LAYER_THICKNESS + TOLERANCE):
    original_first_layer_thickness = dzz[0]
    original_first_layer_type = soil_types[0]

    remainder_thickness = original_first_layer_thickness - TARGET_TOP_LAYER_THICKNESS

    dzz[0] = TARGET_TOP_LAYER_THICKNESS
    dzz.insert(1, round(remainder_thickness, 2))

    soil_types.insert(1, original_first_layer_type)

# Make at least 10 layers of soil, extend the last layer
while len(dzz) < 10:
    dzz.append(dzz[-1])
    soil_types.append(soil_types[-1])

soil = SoilGeorgia(soil_type=soil_types, dz=dzz)

WP = 33.7
HI0 = 0.48
model_os = AquaCropModel(
            sim_start_time=f'{start_date.year}/{start_date.month}/{start_date.day}',
            sim_end_time=f'{end_date.year}/{end_date.month}/{end_date.day}',
            weather_df=prepare_weather(weather_file_path),
            soil=soil,
            crop=Crop('Maize', planting_date=f'{start_date.month}/{start_date.day}', WP=WP, HI0 = HI0),
            initial_water_content=InitialWaterContent(value=['FC']*len(soil.profile.Layer), depth_layer=soil.profile.Layer),
            irrigation_management = irrmethod,
            use_richards = True
        )

model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
# model_results.to_csv('results6.csv')
print(type(model_results))
print(model_results)
print('Actual Yield : ', expobj.crop_yield)