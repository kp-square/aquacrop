from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

weather_file_path = get_filepath('atlanta_climate_daily.txt')

model_os = AquaCropModel(
            sim_start_time=f"{2013}/04/30",
            sim_end_time=f"{2013}/12/20",
            weather_df=prepare_weather(weather_file_path),
            soil=Soil(soil_type='SandyLoam'),
            crop=Crop('Cotton', planting_date='05/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
            irrigation_management=IrrigationManagement(irrigation_method=2),
            step_size='D',
            use_richards=True
        )

model_os.run_model(till_termination=True)
model_results = model_os.get_simulation_results().head()
model_results.to_csv('results3.csv')
print(type(model_results))
print(model_results)