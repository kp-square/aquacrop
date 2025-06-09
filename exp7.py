from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from dataset.dataobjects import SoilType, ExpData
import pickle
from datetime import datetime, timezone, timedelta


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

with open('dataset/experimental_data.pkl', 'rb') as f:
    pickle_data = pickle.load(f)

expobj = pickle_data[0]

weather_file_path = get_filepath('georgia_climate_hourly.txt')
start_date = datetime.fromtimestamp(expobj.start_date / 1e9, tz=timezone.utc)
end_date = datetime.fromtimestamp(expobj.end_date / 1e9, tz=timezone.utc) + timedelta(days=30)

irr_sch = expobj.irr[['DATE', 'irr_depth']]
irr_sch.rename({'DATE':'Date', 'irr_depth':'Depth'}, axis=1, inplace=True)
irrmethod = IrrigationManagement(irrigation_method=3, Schedule=irr_sch)
#irrmethod.Schedule = irr_sch

dzz = []
soil_types = []
prev = 0.0
for typ in expobj.soil_types:
    typ.depth = round(typ.depth, 2)
    typ.soil_type = ''.join(typ.soil_type.split(' '))
    dzz.append(round(typ.depth - prev, 2))
    soil_types.append(typ.soil_type)
    prev = typ.depth

soil = Soil(soil_type=soil_types, dz=dzz)


model_os = AquaCropModel(
            sim_start_time=f'{start_date.year}/{start_date.month}/{start_date.day}',
            sim_end_time=f'{end_date.year}/{end_date.month}/{end_date.day}',
            weather_df=prepare_weather(weather_file_path),
            soil=Soil(soil_type='LoamySand'),
            crop=Crop('Maize', planting_date=f'{start_date.month}/{start_date.day}'),
            initial_water_content=InitialWaterContent(value=['FC']),
            irrigation_management = irrmethod
        )

model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
# model_results.to_csv('results6.csv')
print(type(model_results))
print(model_results)
print('Actual Yield : ', expobj.lint_yield)