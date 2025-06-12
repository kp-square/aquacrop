import numpy as np
import copy
from aquacrop.entities.output import Output
import pickle

# TEMP FOR TROUBLESHOOTING:
from pprint import pprint

from ..entities.totalAvailableWater import TAW
from ..entities.moistureDepletion import Dr
from ..entities.crop import Crop

from ..solution.pre_irrigation import pre_irrigation
from ..solution.irrigation import irrigation
from ..solution.capillary_rise import capillary_rise
from ..solution.germination import germination
from ..solution.growth_stage import growth_stage
from ..solution.canopy_cover import canopy_cover
from ..solution.transpiration import transpiration
from ..solution.transpiration_hourly import  transpiration_daily_setup_update_parameters, transpiration_hourly_setup, transpiration_post_daily
from ..solution.soil_evaporation_hourly import  soil_evaporation_hourly
from ..solution.groundwater_inflow import groundwater_inflow
from ..solution.harvest_index import harvest_index


from ..solution.growing_degree_day import growing_degree_day
from ..solution.drainage import drainage
from ..solution.root_zone_water import root_zone_water
from ..solution.rainfall_partition import rainfall_partition
from ..solution.check_groundwater_table import check_groundwater_table
from ..solution.soil_evaporation import soil_evaporation
from ..solution.root_development import root_development
from ..solution.infiltration import infiltration
from ..solution.HIref_current_day import HIref_current_day
from ..solution.biomass_accumulation import biomass_accumulation
from ..utils.richards.plots import plot_crop_simulation_data
from ..utils.richards.richards_utils import irrigation_dissociation
from ..optim.dataobjects import InputState, OutputState

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Important: classes are only imported when types are checked, not in production.
    from numpy import ndarray
    from aquacrop.entities.clockStruct import ClockStruct
    from aquacrop.entities.initParamVariables import InitialCondition
    from aquacrop.entities.paramStruct import ParamStruct
    from aquacrop.entities.output import Output
    from aquacrop.entities.crop import Crop

from ..utils.richards.richards_eqn_solver import RichardEquationSolver
def solution_single_time_step_richards(
    init_cond: "InitialCondition",
    param_struct: "ParamStruct",
    clock_struct: "ClockStruct",
    weather_step: "ndarray",
    outputs:"Output",
) ->  Tuple["InitialCondition", "ParamStruct","Output"]:
    """
    Function to perform AquaCrop solution for a single time step

    Arguments:

        init_cond (InitialCondition):  containing current variables+counters

        param_struct (ParamStruct):  contains model paramaters

        clock_struct (ClockStruct):  model time paramaters

        weather_step (numpy.ndarray):  contains precipitation,ET,temp_max,temp_min for current day

        outputs (Output):  object to store outputs

    Returns:

        NewCond (InitialCondition):  containing updated simulation variables+counters

        param_struct (ParamStruct):  contains model paramaters

        outputs (Output):  object to store outputs

    """

    # Unpack structures
    Soil = param_struct.Soil
    CO2 = param_struct.CO2
    precipitation = weather_step[:,1]
    temp_mean = weather_step[:,0]
    temp_min = min(temp_mean)
    temp_max = max(temp_mean)
    et0 = weather_step[:,2]
    precipitation_daily = sum(precipitation)

    # Store initial conditions in structure for updating %%
    NewCond = init_cond
    if param_struct.water_table == 1:
        GroundWater = param_struct.z_gw[clock_struct.time_step_counter]
    else:
        GroundWater = 0

    # Check if growing season is active on current time step %%
    if clock_struct.season_counter >= 0:
        # Check if in growing season
        CurrentDate = clock_struct.step_start_time
        planting_date = clock_struct.planting_dates[clock_struct.season_counter]
        harvest_date = clock_struct.harvest_dates[clock_struct.season_counter]

        if (
            (planting_date <= CurrentDate)
            and (harvest_date >= CurrentDate)
            and (NewCond.crop_mature is False)
            and (NewCond.crop_dead is False)
        ):
            growing_season = True
        else:
            growing_season = False
        # Assign crop, irrigation management, and field management structures
        Crop_ = param_struct.Seasonal_Crop_List[clock_struct.season_counter]
        Crop_Name = param_struct.CropChoices[clock_struct.season_counter]
        IrrMngt = param_struct.IrrMngt

        if growing_season is True:
            FieldMngt = param_struct.FieldMngt
        else:
            FieldMngt = param_struct.FallowFieldMngt

    else:
        # Not yet reached start of first growing season
        growing_season = False
        # Assign crop, irrigation management, and field management structures
        # Assign first crop as filler crop
        Crop_ = param_struct.Fallow_Crop
        Crop_Name = "fallow"

        Crop_.Aer = 5
        Crop_.Zmin = 0.3
        IrrMngt = param_struct.FallowIrrMngt
        FieldMngt = param_struct.FallowFieldMngt

    # Increment time counters %%
    if growing_season is True:
        # Calendar days after planting
        NewCond.dap = NewCond.dap + 1
        # Growing degree days after planting
        gdd = growing_degree_day(
            Crop_.GDDmethod, Crop_.Tupp, Crop_.Tbase, temp_max, temp_min
        )

        # Update cumulative gdd counter
        NewCond.gdd = gdd
        NewCond.gdd_cum = NewCond.gdd_cum + gdd

        NewCond.growing_season = True
    else:
        NewCond.growing_season = False

        # Calendar days after planting
        NewCond.dap = 0
        # Growing degree days after planting
        gdd = 0.3
        NewCond.gdd_cum = 0

    # save current timestep counter
    NewCond.time_step_counter = clock_struct.time_step_counter
    NewCond.precipitation = precipitation
    NewCond.temp_max = temp_max
    NewCond.temp_min = temp_min
    NewCond.et0 = et0

    crop = Crop_
    total_water_begin = sum(NewCond.th * Soil.profile.dz) * 1000
    init_root_z = NewCond.z_root
    # Run simulations %%
    # 1. Check for groundwater table (Run once daily before hourly loop)
    NewCond.th_fc_Adj, NewCond.wt_in_soil, NewCond.z_gw = check_groundwater_table(
        Soil.Profile,
        NewCond.z_gw,
        NewCond.th,
        NewCond.th_fc_Adj,
        param_struct.water_table,
        GroundWater
    )

    # Germination Check (Run once daily)
    NewCond = germination(
        NewCond,
        Soil.z_germ,
        Soil.Profile,
        crop.GermThr,
        crop.PlantMethod,
        gdd,
        growing_season,
    )

    # 2. Root development (Run once daily)
    NewCond.z_root, NewCond.r_cor = root_development(
        crop,
        Soil.Profile,
        NewCond.dap,
        NewCond.z_root,
        NewCond.delayed_cds,
        NewCond.gdd_cum,
        NewCond.delayed_gdds,
        NewCond.tr_ratio,
        NewCond.th,
        NewCond.canopy_cover,
        NewCond.canopy_cover_ns,
        NewCond.germination,
        NewCond.r_cor,
        NewCond.t_pot,
        NewCond.z_gw,
        gdd,
        growing_season,
        param_struct.water_table,
    )

    # Growth Stage Update (Run once daily)
    NewCond = growth_stage(crop, NewCond, growing_season)

    # Canopy Cover Update (Run once daily)
    # Note: Needs daily et0 instead of hourly et0
    et0_daily = sum(et0)
    NewCond = canopy_cover(
        crop, Soil.Profile, Soil.z_top, NewCond, gdd, et0_daily, growing_season
    )

    # 3. Pre-irrigation (Run once daily)
    # Note: Add PreIrr to the irrigation amount of the first hour of the hourly loop.
    NewCond, PreIrr = pre_irrigation(
        Soil.Profile, crop, NewCond, growing_season, IrrMngt
    )

    # Daily Transpiration Setup (Run once daily to get daily parameters)
    Kcb_NS, Kcb, KsCold, NewCond = transpiration_daily_setup_update_parameters(
        Soil.Profile,
        Soil.nComp,
        crop,
        NewCond,
        CO2,
        growing_season,
        gdd,
    )

    Runoff = 0.0

    NewCond.depletion, NewCond.taw, NewCond.irr_cum, Irr = irrigation(
        IrrMngt.irrigation_method,
        IrrMngt.SMT,
        IrrMngt.AppEff,
        IrrMngt.MaxIrr,
        IrrMngt.IrrInterval,
        IrrMngt.Schedule,
        IrrMngt.depth,
        IrrMngt.MaxIrrSeason,
        NewCond.growth_stage,
        NewCond.irr_cum,
        NewCond.e_pot,
        NewCond.t_pot,
        NewCond.z_root,
        NewCond.th,
        NewCond.dap,
        NewCond.time_step_counter,
        crop,
        Soil.Profile,
        Soil.z_top,
        growing_season,
        precipitation_daily,
        Runoff
    )


    # Initialize daily accumulators

    Tr_daily = 0
    TrPot_NS_daily = 0
    TrPot_daily = 0
    Es_daily = 0

    EsPot_daily = 0
    DeepPerc_daily = 0
    Runoff_daily = 0
    Infl_daily = 0
    Irr_applied_daily = 0  # Keep track of irrigation actually applied

    # Store the state at the beginning of the day (start of the hourly loop)
    PrevCond = copy.deepcopy(NewCond)
    instate = InputState(precipitation, et0, temp_min, temp_max, NewCond.gdd, NewCond.dap, init_root_z, NewCond.canopy_cover, NewCond.th, NewCond.biomass)


    Infl = 0.0
    hourly_irrigation = irrigation_dissociation(Irr)
    solver = RichardEquationSolver(Soil.profile, PrevCond, time_step='h')
    # --- Hourly Loop --- #
    for hour in range(24):
        et0_hr = et0[hour]
        precipitation_hr = precipitation[hour]
        '''
        NewCond_Depletion (float): soil water depletion
        NewCond_TAW (float): total available water
        NewCond_IrrCum (float): total irrigation applied so far
        Irr (float): Irrigation applied on current hour mm
        '''

        # PrevCond = NewCond
        # Updates the water content in each layer
        # TrAct_ = Total actual transpiration
        # TrPot_NS_ = Total potential transpiration with No stress
        # TrPot_ = Total potential transpiration in ideal condition
        # NewCond = NewCond.th contains the updated water in each compartment
        # IrrNet_ = irrigation water to be applied to balance to make each compartment have critical water content
        TrAct_, TrPot_NS_, TrPot_, NewCond, IrrNet_ = transpiration_hourly_setup(
                                                    Soil.Profile,
                                                    Soil.nComp,
                                                    Soil.z_top,
                                                    crop,
                                                    IrrMngt.irrigation_method,
                                                    IrrMngt.NetIrrSMT,
                                                    NewCond,
                                                    et0_hr,
                                                    Kcb,
                                                    Kcb_NS,
                                                    KsCold,
                                                    growing_season
                                                )


        Tr_daily += TrAct_
        TrPot_NS_daily += TrPot_NS_
        TrPot_daily += TrPot_
        # if irrigation_method  == 4, IrrNet_ != 0 else IrrNet_ = 0
        Irr_applied_daily += IrrNet_


        '''
        NewCond_Epot (float): Potential surface evaporation current day
        NewCond_th (ndarray): updated soil water content
        NewCond_Stage2 (bool): stage 2 soil evaporation
        NewCond_Wstage2 (float): stage 2 soil evaporation 
        NewCond_Wsurf (float): updated surface water content
        NewCond_SurfaceStorage (float): updated surface storage
        NewCond_EvapZ (float): updated evaporation layer depth
        EsAct (float): Actual surface evaporation current day
        EsPot (float): Potential surface evaporation current day
        '''
        (
            NewCond.e_pot,
            NewCond.th,
            NewCond.stage2,
            NewCond.w_stage_2,
            NewCond.w_surf,
            NewCond.surface_storage,
            NewCond.evap_z,
            Es,
            EsPot,
        ) = soil_evaporation_hourly(
            # clock_struct.evap_time_steps,
            clock_struct.sim_off_season,
            clock_struct.time_step_counter,
            Soil.Profile,
            Soil.evap_z_min,
            Soil.evap_z_max,
            Soil.rew,
            Soil.kex,
            Soil.fwcc,
            Soil.f_wrel_exp,
            Soil.f_evap,
            crop.CalendarType,
            crop.Senescence,
            IrrMngt.irrigation_method,
            IrrMngt.WetSurf,
            FieldMngt.mulches,
            FieldMngt.f_mulch,
            FieldMngt.mulch_pct,
            NewCond.dap,
            NewCond.w_surf,
            NewCond.evap_z,
            NewCond.stage2,
            NewCond.th,
            NewCond.delayed_cds,
            NewCond.gdd_cum,
            NewCond.delayed_gdds,
            NewCond.ccx_w,
            NewCond.canopy_cover_adj,
            NewCond.ccx_act,
            NewCond.canopy_cover,
            NewCond.premat_senes,
            NewCond.surface_storage,
            NewCond.w_stage_2,
            #NewCond.e_pot,
            et0_hr,
            Infl,
            precipitation_hr,
            Irr,
            growing_season,
        )

        Es_daily += Es
        EsPot_daily += EsPot
        # NewCond.th has the volumetric water content for each compartment

        converged, new_th, DeepPerc, Runoff, Infl, FluxOut, _, _ = solver.solve(hour, NewCond,
                                                                                    hourly_irrigation[hour],
                                                                                    precipitation_hr)
        if converged:
            NewCond.th = new_th
            DeepPerc_daily += DeepPerc*1000
            Runoff_daily += Runoff*1000
            Infl_daily += Infl*1000


    NewCond = transpiration_post_daily(NewCond, Tr_daily, TrPot_daily, growing_season)

    # 15. Reference harvest index
    (NewCond.hi_ref, NewCond.yield_form, NewCond.pct_lag_phase) = HIref_current_day( # ,NewCond.HIfinal
        NewCond.hi_ref,
        NewCond.HIfinal,
        NewCond.dap,
        NewCond.delayed_cds,
        NewCond.yield_form,
        NewCond.pct_lag_phase,
        NewCond.canopy_cover,
        NewCond.cc_prev,
        NewCond.ccx_w,
        crop,
        growing_season,
    )

    # 16. Biomass accumulation
    (NewCond.biomass, NewCond.biomass_ns) = biomass_accumulation(
        crop,
        NewCond.dap,
        NewCond.delayed_cds,
        NewCond.hi_ref,
        NewCond.pct_lag_phase,
        NewCond.biomass,
        NewCond.biomass_ns,
        Tr_daily,
        TrPot_NS_daily,
        et0_daily,
        growing_season,
    )

    # 17. Harvest index
    NewCond = harvest_index(
        Soil.Profile, Soil.z_top, crop, NewCond, et0_daily, temp_max, temp_min, growing_season
    )

    # 18. Yield potential
    NewCond.YieldPot = (NewCond.biomass_ns / 100) * NewCond.harvest_index

    # 19. Crop yield_ (dry and fresh)
    if growing_season is True:
        # Calculate crop yield_ (tonne/ha)
        NewCond.DryYield = (NewCond.biomass / 100) * NewCond.harvest_index_adj
        NewCond.FreshYield = NewCond.DryYield / (crop.YldWC / 100)
        # print( clock_struct.time_step_counter,(NewCond.biomass/100),NewCond.harvest_index_adj)
        # Check if crop has reached maturity
        if ((crop.CalendarType == 1) and (NewCond.dap >= crop.Maturity)) or (
            (crop.CalendarType == 2) and (NewCond.gdd_cum >= crop.Maturity)
        ):
            # Crop has reached maturity
            NewCond.crop_mature = True

    elif growing_season is False:
        # Crop yield_ is zero outside of growing season
        NewCond.DryYield = 0
        NewCond.FreshYield = 0

    # 20. Root zone water
    _TAW = TAW()
    _water_root_depletion = Dr()
    # thRZ = RootZoneWater()

    Wr, _water_root_depletion.Zt, _water_root_depletion.Rz, _TAW.Zt, _TAW.Rz, _, _, _, _, _, _ = root_zone_water(
        Soil.Profile,
        float(NewCond.z_root),
        NewCond.th,
        Soil.z_top,
        float(crop.Zmin),
        crop.Aer,
    )

    # 21. Update net irrigation to add any pre irrigation
    IrrNet = Irr_applied_daily + PreIrr
    NewCond.irr_net_cum = NewCond.irr_net_cum + PreIrr

    # Update model outputs %%
    row_day = clock_struct.time_step_counter
    row_gs = clock_struct.season_counter
    # Irrigation
    if growing_season is True:
        if IrrMngt.irrigation_method == 4:
            # Net irrigation
            IrrDay = IrrNet
            IrrTot = NewCond.irr_net_cum
        else:
            # Irrigation
            IrrDay = Irr
            IrrTot = NewCond.irr_cum

    else:
        IrrDay = 0
        IrrTot = 0

        NewCond.depletion = _water_root_depletion.Rz
        NewCond.taw = _TAW.Rz

    outstate = OutputState(NewCond.canopy_cover - PrevCond.canopy_cover, NewCond.z_root - init_root_z, Es_daily, Tr_daily,
                           NewCond.biomass - PrevCond.biomass, NewCond.DryYield - PrevCond.DryYield,
                           NewCond.th - PrevCond.th, DeepPerc_daily, Runoff_daily)

    outputs.instates.append(instate)
    outputs.outstates.append(outstate)
    # Water contents
    outputs.water_storage[row_day, :3] = np.array(
        [clock_struct.time_step_counter, growing_season, NewCond.dap]
    )
    outputs.water_storage[row_day, 3:] = NewCond.th

    # Water fluxes
    # print(f'Saving NewCond.z_gw to outputs: {NewCond.z_gw}')

    # -- we ignore groundwater inflow for now -- #
    CR = 0.0
    GwIn = 0.0
    # --------------
    total_water_end = sum(NewCond.th * Soil.profile.dz) * 1000
    outputs.water_flux.loc[row_day] = [
        clock_struct.time_step_counter,
        clock_struct.season_counter,
        NewCond.dap,
        Wr,
        NewCond.z_gw,
        NewCond.surface_storage,
        IrrDay,
        sum(precipitation),
        Infl_daily,
        Runoff_daily,
        DeepPerc_daily,
        CR,
        GwIn,
        Es_daily,
        EsPot_daily,
        Tr_daily,
        TrPot_daily,
        total_water_begin,
        total_water_begin + Infl_daily - Es_daily - Tr_daily - DeepPerc_daily - total_water_end
    ]

    # if row_day == 174:
    #     desc = outputs.water_flux.describe()
    #     plot_crop_simulation_data(outputs.water_flux, f'hourly_sim_plot.png')

    # Crop growth
    outputs.crop_growth[row_day, :] = [
        clock_struct.time_step_counter,
        clock_struct.season_counter,
        NewCond.dap,
        gdd,
        NewCond.gdd_cum,
        NewCond.z_root,
        NewCond.canopy_cover,
        NewCond.canopy_cover_ns,
        NewCond.biomass,
        NewCond.biomass_ns,
        NewCond.harvest_index,
        NewCond.harvest_index_adj,
        NewCond.DryYield,
        NewCond.FreshYield,
        NewCond.YieldPot,
    ]

    # Final output (if at end of growing season)
    if clock_struct.season_counter > -1:
        if (
            (NewCond.crop_mature is True)
            or (NewCond.crop_dead is True)
            or (
                clock_struct.harvest_dates[clock_struct.season_counter]
                == clock_struct.step_end_time
            )
        ) and (NewCond.harvest_flag is False):
            desc = outputs.water_flux.describe()

            total_rainfall = outputs.water_flux['precipitation'].sum()
            total_infl = outputs.water_flux['Infl'].sum()
            total_runoff = outputs.water_flux['Runoff'].sum()
            total_irr = outputs.water_flux['IrrDay'].sum()
            total_es = outputs.water_flux['Es'].sum()
            total_tr  = outputs.water_flux['Tr'].sum()
            total_dp = outputs.water_flux['DeepPerc'].sum()
            total_err = outputs.water_flux['balance'].abs().sum()

            # Store final outputs
            outputs.final_stats.loc[row_gs] = [
                clock_struct.season_counter,
                Crop_Name,
                clock_struct.step_end_time,
                clock_struct.time_step_counter,
                NewCond.DryYield,
                NewCond.FreshYield,
                NewCond.YieldPot,
                IrrTot,
                total_rainfall,
                total_es,
                total_tr,
                total_dp,
                total_runoff,
                total_infl,
                total_err
            ]
            # try:
            #     with open('aquacrop/optim/train_data.pkl', 'wb') as f:
            #         obj = (instate, outstate)
            #         pickle.dump((hash(obj), obj), f)
            # except:
            #     print('error')



            # Set harvest flag
            NewCond.harvest_flag = True

    return NewCond, param_struct, outputs
