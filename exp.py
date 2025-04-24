from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath

weather_file_path = get_filepath('atlanta_climate_daily.txt')
model_os = AquaCropModel(
            sim_start_time=f"{2008}/04/30",
            sim_end_time=f"{2008}/12/20",
            weather_df=prepare_weather(weather_file_path),
            soil=Soil(soil_type='SandyLoam'),
            crop=Crop('Cotton', planting_date='05/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
        )
model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
model_results.to_csv('results2.csv')
print(type(model_results))
print(model_results)