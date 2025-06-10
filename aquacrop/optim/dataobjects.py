class InputState:
    def __init__(self, precip, refET, temp_min, temp_max, gdd, dap, root_length, canopy_cover, moisture, biomass):
        self.precip = precip # daily precipitation (24h)
        self.refET = refET # reference ET0
        self.temp_min = temp_min # min temperature
        self.temp_max = temp_max # max temperature
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
