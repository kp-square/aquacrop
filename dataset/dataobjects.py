import pandas as pd
import numpy as np

class SoilType:
    def __init__(self, year, crop_type, sample_id, soil_type, depth):
        self.sirp_id = sample_id
        self.soil_type = soil_type
        self.depth = depth


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
        self.soil_types = None #Soil Profile with different layers of soil identified by sirp_id


class InputState:
    def __init__(self, precip, refET, temp_mean, gdd, dap, root_length, canopy_cover, moisture, biomass):
        self.precip = precip # daily precipitation (24h)
        self.refET = refET # reference ET0
        self.temp_mean = temp_mean # mean temperature
        self.gdd = gdd # growth degree days
        self.day_of_growth = dap # day of planting
        self.root_length = root_length
        self.canopy_cover = canopy_cover
        self.theta = moisture # moisture content in each compartments
        self.biomass = biomass

class OutputState:
    def __init__(self, del_canopy_cover, del_root_length, evaporation, transpiration, del_biomass, yield_val, del_theta, deep_perc, runoff):
        self.del_canopy_cover = del_canopy_cover
        self.del_root_length = del_root_length
        self.evaporation = evaporation
        self.transpiration = transpiration
        self.del_biomass = del_biomass
        self.yield_val = yield_val
        self.del_theta = del_theta
        self.deep_perc = deep_perc
        self.runoff = runoff
