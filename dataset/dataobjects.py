import pandas as pd
import numpy as np

class SoilType:
    def __init__(self, year, crop_type, sample_id, soil_type, depth):
        self.sirp_id = sample_id
        self.soil_type = soil_type
        self.depth = depth


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
        self.soil_types = None #Soil Profile with different layers of soil identified by sirp_id