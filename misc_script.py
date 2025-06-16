# import pandas as pd
# import os
# from dataset.dataobjects import SoilType, ExpData
# import pandas, pickle
#
# def main():
#     with open('dataset/experimental_data.pkl', 'rb') as f:
#         pickle_data = pickle.load(f)
#     # crop_type, year, treatment_id, sirp_id
#     data = {'crop_type':[], 'year':[], 'treatment_id':[], 'sirp_id':[]}
#     for obj in pickle_data:
#         data['crop_type'].append(obj.crop_type)
#         data['year'].append(obj.year)
#         data['treatment_id'].append(obj.treatment_id)
#         data['sirp_id'].append(obj.sirp_id)
#
#     df = pd.DataFrame.from_dict(data)
#     df.to_csv('crop_metadata.csv')
#
#
# def submit_slurm_job(WP, HI0, run_count):
#     df = pd.read_csv('crop_metadata.csv')
#     slurm_script = '''
#         #!/bin/bash
#         #SBATCH --nodes=1
#         #SBATCH --tasks=1
#         #SBATCH --cpus=1
#         #SBATCH --RAM=4g
#         #SBATCH --time=72:00:00
#
#         module load anaconda/3
#         source activate drl
#         python simulation_script.py --crop_type {0} --year {1} --treatment_id {2}, --sirp_id {3}, hourly {4}, use_richards {5}, WP {6}, HI0 {7}, run_count {8}
#
#     '''
#     filter_df = df[(df['crop_type']=='corn' & (df['year'] == 2018 | df['year'] == 2019))]
#     for row, _ in filter_df.iterrows():
#         updated_script = slurm_script.format(row.get('crop_type'), row.get('year'), row.get('treatment_id'), row.get('sirp_id'), True, True, WP, HI0, run_count)
#         os.execv('sbatch', updated_script)

import pandas as pd
import pickle

import numpy as np
from dataset.dataobjects import ExpData, SoilType

def all_soil_types():
    with open('dataset/experimental_data.pkl', 'rb') as f:
        data = pickle.load(f)

    alltypes = set()
    for obj in data:
        types = obj.soil_types
        for typ in types:
            alltypes.add(typ.soil_type)

    print(list(alltypes))
def calculate_soil_hydraulic_properties( Sand, Clay, OrgMat, DF=1):
    """
    Function to calculate soil hydraulic properties, given textural inputs.
    Calculations use pedotransfer function equations described in Saxton and Rawls (2006)

    Input Parameters:
    Sand: Percentage of sand in the soil
    Clay: Percentage of clay in the soil
    OrgMat: Percentage of organic matter in the soil
    DF: Density Factor, related to bulk density

    Returns:
        th_wp: Water content at permanent wilting point. It's the volumetric water content at which
                plants can no longer extract water from the soil and will permanently wilt (wither).
        th_fc: Water content at field capacity. Volumetric water content of the soil after it has been saturated
                and allowed to drain freely for a day or two. Represents upper limit of water available for plants.
        th_s: Water content at saturation. Volumetric water content when all soil pores are filled with water.
        Ksat: Saturated hydraulic conductivity. Represents the ease with which water moves through a saturated soil.
                Expressed in mm/day.
    """

    # do calculations

    # Water content at permanent wilting point
    Pred_thWP = (
            -(0.024 * Sand)
            + (0.487 * Clay)
            + (0.006 * OrgMat)
            + (0.005 * Sand * OrgMat)
            - (0.013 * Clay * OrgMat)
            + (0.068 * Sand * Clay)
            + 0.031
    )

    th_wp = Pred_thWP + (0.14 * Pred_thWP) - 0.02

    # Water content at field capacity and saturation
    Pred_thFC = (
            -(0.251 * Sand)
            + (0.195 * Clay)
            + (0.011 * OrgMat)
            + (0.006 * Sand * OrgMat)
            - (0.027 * Clay * OrgMat)
            + (0.452 * Sand * Clay)
            + 0.299
    )

    PredAdj_thFC = Pred_thFC + (
            (1.283 * (np.power(Pred_thFC, 2))) - (0.374 * Pred_thFC) - 0.015
    )

    Pred_thS33 = (
            (0.278 * Sand)
            + (0.034 * Clay)
            + (0.022 * OrgMat)
            - (0.018 * Sand * OrgMat)
            - (0.027 * Clay * OrgMat)
            - (0.584 * Sand * Clay)
            + 0.078
    )

    PredAdj_thS33 = Pred_thS33 + ((0.636 * Pred_thS33) - 0.107)
    Pred_thS = (PredAdj_thFC + PredAdj_thS33) + ((-0.097 * Sand) + 0.043)

    pN = (1 - Pred_thS) * 2.65
    pDF = pN * DF
    PorosComp = (1 - (pDF / 2.65)) - (1 - (pN / 2.65))
    PorosCompOM = 1 - (pDF / 2.65)

    DensAdj_thFC = PredAdj_thFC + (0.2 * PorosComp)
    DensAdj_thS = PorosCompOM

    th_fc = DensAdj_thFC
    th_s = DensAdj_thS

    # Saturated hydraulic conductivity (mm/day)
    lmbda = 1 / ((np.log(1500) - np.log(33)) / (np.log(th_fc) - np.log(th_wp)))
    Ksat = (1930 * (th_s - th_fc) ** (3 - lmbda)) * 24

    # Water content at air dry
    th_dry = th_wp / 2

    # round values
    th_dry = round(10000 * th_dry) / 10_000
    th_wp = round(1000 * th_wp) / 1000
    th_fc = round(1000 * th_fc) / 1000
    th_s = round(1000 * th_s) / 1000
    Ksat = round(10 * Ksat) / 10

    return th_wp, th_fc, th_s, Ksat

all_soil_types()
print('th_wp', 'th_fc', 'th_s', 'Ksat')
print('Loamy Sand', calculate_soil_hydraulic_properties(0.82, 0.08, 0.01))
print('Sandy Loam', calculate_soil_hydraulic_properties(0.81, 0.13, 0.01))
print('Sand', calculate_soil_hydraulic_properties(0.936, 0.024, 0.01))
print('Sandy Clay Loam', calculate_soil_hydraulic_properties(0.71, 0.28, 0.01))
print('Loamy Fine Sand', calculate_soil_hydraulic_properties(0.86, 0.06, 0.01))
print('Fine Sand', calculate_soil_hydraulic_properties(0.89, 0.045, 0.01))