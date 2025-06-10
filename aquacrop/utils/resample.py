import pandas as pd

def resample_from_hourly_to_daily(data):
    temp = data.set_index('Date')
    daily_data = temp.resample('D').agg({
        'MeanTemp': ['min', 'max'],
        'Precipitation': 'sum',
        'ReferenceET': 'sum'
    })
    daily_data.columns = ['MinTemp', 'MaxTemp', 'Precipitation', 'ReferenceET']
    daily_data.reset_index(inplace=True)
    return daily_data