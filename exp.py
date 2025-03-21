from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath

weather_file_path = get_filepath('tunis_climate.txt')
model_os = AquaCropModel(
            sim_start_time=f"{1979}/10/01",
            sim_end_time=f"{1985}/05/30",
            weather_df=prepare_weather(weather_file_path),
            soil=Soil(soil_type='SandyLoam'),
            crop=Crop('Cotton', planting_date='10/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
        )
model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
model_results.to_csv('results1.csv')
print(type(model_results))
print(model_results)