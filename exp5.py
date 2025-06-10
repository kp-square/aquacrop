from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath

weather_file_path = get_filepath('georgia_climate_hourly.txt')

model_os = AquaCropModel(
            sim_start_time=f"{2021}/04/30",
            sim_end_time=f"{2021}/12/20",
            weather_df=prepare_weather(weather_file_path, hourly=True),
            soil=Soil(soil_type='LoamySand'),
            crop=Crop('Cotton', planting_date='05/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
            step_size='H',
            use_richards=True
        )

model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
model_results.to_csv('results5.csv')
print(type(model_results))
print(model_results)