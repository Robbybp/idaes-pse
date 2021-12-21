#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
""" Tests for the example of MHE for Rodrigo's cstr
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.examples.cstr_rodrigo.mhe_main import (
            setup_estimator,
            solve_first_esitmation_NLP,
            run_iterations,
            )
from idaes.apps.caprese.dynamic_builder import (
            DynamicSim
            )
from idaes.apps.caprese.data_manager import(
            PlantDataManager,
            EstimatorDataManager,
            )

class TestMHEMain(object):

    @pytest.mark.unit
    def test_setup_estimator(self):
        mhe, plant_data, estimator_data = setup_estimator(mhe_horizon=10,
                                                          mhe_ntfe=10,
                                                          mhe_ntcp=2,
                                                          plant_ntfe=4,
                                                          plant_ntcp=2,
                                                          sample_time=2.0)

        pmod = mhe.plant.mod
        var_value_dict = {"Ca[0]": 0.0192,
                           "k[0]": 1596.67471,
                           "Tjinb[0]": 250.0,
                           "Tall[0,T]": 384.0,
                           "Tall[0,Tj]": 371.0,
                           }
        for varstr_t0 in var_value_dict.keys():
            cuid = pyo.ComponentUID(varstr_t0)
            comp = cuid.find_component_on(pmod)
            assert comp.value == pytest.approx(var_value_dict[varstr_t0],
                                               1e-3,
                                               )

        emod = mhe.estimator.mod
        var_value_dict = {"Ca[*]" : 0.02099,
                          "k[*]" : 1333.27682,
                          "Tjinb[*]" : 235.55240,
                          "Cadot[*]" : 0.0,
                          "Tall[*,T]" : 381.13357,
                          "Tall[*,Tj]" : 367.30007,
                          "Talldot[*,T]" : 0.0,
                          "Talldot[*,Tj]" : 0.0,}
        for varstr_slice in var_value_dict.keys():
            cuid = pyo.ComponentUID(varstr_slice)
            for comp in cuid.find_component_on(emod).values():
                assert comp.value == pytest.approx(
                    var_value_dict[varstr_slice],
                    1e-3,
                    )

        actmea_block = mhe.estimator.ACTUALMEASUREMENT_BLOCK
        assert all(mea_comp.value == pytest.approx(var_value_dict["Tall[*,T]"],
                                                   1e-3,)
                   for mea_comp in actmea_block[0].var[:])
        assert all(mea_comp.value == pytest.approx(var_value_dict["Ca[*]"],
                                                   1e-3,)
                   for mea_comp in actmea_block[1].var[:])

        assert hasattr(mhe.estimator, 'noise_minimize_objective')

        assert type(mhe) is DynamicSim
        assert type(plant_data) is PlantDataManager
        assert type(estimator_data) is EstimatorDataManager

    @pytest.mark.component
    def test_solve_first_estimation_NLP(self):
        mhe, plant_data, estimator_data = setup_estimator(
                                            mhe_horizon=10,
                                            mhe_ntfe=10,
                                            mhe_ntcp=2,
                                            plant_ntfe=4,
                                            plant_ntcp=2,
                                            sample_time=2.0
                                            )

        mhe, plant_data, estimator_data = solve_first_esitmation_NLP(
                                            mhe,
                                            plant_data,
                                            estimator_data
                                            )

        pmod = mhe.plant.mod
        pmod_var_value_dict = {"Ca[2.0]": 0.01919,
                               "Tall[2.0,T]": 384.00519,
                               "Tall[2.0,Tj]": 371.27157,
                               "k[2.0]": 1597.19180,
                               "Tjinb[2.0]": 250.0,
                               }
        for comp in pmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_pmod = cuid.find_component_on(pmod)
            assert comp_pmod.value == pytest.approx(pmod_var_value_dict[comp],
                                                    1e-3)

        plant_df = plant_data.plant_df
        assert len(plant_df.index) == 9
        assert plant_df["mod.Ca[*]"][2] == pytest.approx(
                                        pmod_var_value_dict["Ca[2.0]"],
                                        1e-3)
        assert plant_df["mod.Tall[*,T]"][2] == pytest.approx(
                                        pmod_var_value_dict["Tall[2.0,T]"],
                                        1e-3)
        assert plant_df["mod.Tall[*,Tj]"][2] == pytest.approx(
                                        pmod_var_value_dict["Tall[2.0,Tj]"],
                                        1e-3)
        assert plant_df["mod.Tjinb[*]"][2] == pytest.approx(
                                        pmod_var_value_dict["Tjinb[2.0]"],
                                        1e-3)

        actmea_block = mhe.estimator.ACTUALMEASUREMENT_BLOCK
        sample_point = mhe.estimator.SAMPLEPOINT_SET
        assert actmea_block[0].var[sample_point.last()].value == pytest.approx(
                                            pmod_var_value_dict["Tall[2.0,T]"],
                                            1e-3
                                            )
        assert actmea_block[1].var[sample_point.last()].value == pytest.approx(
                                            pmod_var_value_dict["Ca[2.0]"],
                                            1e-3
                                            )

        emod = mhe.estimator.mod
        check_control_t_list = [t for t in emod.t
                                if t > sample_point.at(-2)
                                and t <= sample_point.last()]
        assert all(emod.Tjinb[t].is_fixed() for t in check_control_t_list)
        assert all(emod.Tjinb[t].value  == pytest.approx(
                                            pmod_var_value_dict["Tjinb[2.0]"],
                                            1e-3
                                            )
                   for t in check_control_t_list)

        emod_var_value_dict = {"Ca[10]": 0.01921,
                               "Tall[10,T]": 384.00519,
                               "Tall[10,Tj]": 371.261154,
                               "k[10]": 1597.19171,
                               }
        for comp in emod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_pmod = cuid.find_component_on(emod)
            assert comp_pmod.value == pytest.approx(emod_var_value_dict[comp],
                                                    1e-3)

        estimator_df = estimator_data.estimator_df
        assert len(estimator_df.index) == 1
        assert estimator_df["mod.Ca[*]"][2] == pytest.approx(
                                        emod_var_value_dict["Ca[10]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,T]"][2] == pytest.approx(
                                        emod_var_value_dict["Tall[10,T]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,Tj]"][2] == pytest.approx(
                                        emod_var_value_dict["Tall[10,Tj]"],
                                        1e-3)

    @pytest.mark.component
    def test_run_iterations(self):
        mhe, plant_data, estimator_data = setup_estimator(
                                            mhe_horizon=10,
                                            mhe_ntfe=10,
                                            mhe_ntcp=2,
                                            plant_ntfe=4,
                                            plant_ntcp=2,
                                            sample_time=2.0
                                            )

        mhe, plant_data, estimator_data = solve_first_esitmation_NLP(
                                            mhe,
                                            plant_data,
                                            estimator_data
                                            )

        mhe, plant_data, estimator_data = run_iterations(
                                            mhe,
                                            plant_data,
                                            estimator_data,
                                            iterations=2,
                                            noise_info=None,
                                            plot_results=False,
                                            )

        pmod = mhe.plant.mod
        pmod_var_value_dict = {"Ca[0.0]": 0.018655,
                                "Tall[0.0,T]": 384.930409,
                                "Tall[0.0,Tj]": 372.581354,
                                "k[0.0]": 1691.919933,
                                "Tjinb[0.0]": 260.0,
                                "Ca[2.0]": 0.018101,
                                "Tall[2.0,T]": 385.913247,
                                "Tall[2.0,Tj]": 373.945749,
                                "k[2.0]": 1798.164102,
                                "Tjinb[2.0]": 260.0,
                                }
        for comp in pmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_pmod = cuid.find_component_on(pmod)
            assert comp_pmod.value == pytest.approx(pmod_var_value_dict[comp],
                                                    1e-3)

        plant_df = plant_data.plant_df
        assert len(plant_df.index) == 25
        assert plant_df["mod.Ca[*]"][6] == pytest.approx(
                                        pmod_var_value_dict["Ca[2.0]"],
                                        1e-3)
        assert plant_df["mod.Tall[*,T]"][6] == pytest.approx(
                                        pmod_var_value_dict["Tall[2.0,T]"],
                                        1e-3)
        assert plant_df["mod.Tall[*,Tj]"][6] == pytest.approx(
                                        pmod_var_value_dict["Tall[2.0,Tj]"],
                                        1e-3)
        assert plant_df["mod.Tjinb[*]"][6] == pytest.approx(
                                        pmod_var_value_dict["Tjinb[2.0]"],
                                        1e-3)

        actmea_block = mhe.estimator.ACTUALMEASUREMENT_BLOCK
        assert actmea_block[0].var[8.0].value == pytest.approx(
                                            pmod_var_value_dict["Tall[0.0,T]"],
                                            1e-3
                                            )
        assert actmea_block[1].var[8.0].value == pytest.approx(
                                            pmod_var_value_dict["Ca[0.0]"],
                                            1e-3
                                            )
        assert actmea_block[0].var[10.0].value == pytest.approx(
                                            pmod_var_value_dict["Tall[2.0,T]"],
                                            1e-3
                                            )
        assert actmea_block[1].var[10.0].value == pytest.approx(
                                            pmod_var_value_dict["Ca[2.0]"],
                                            1e-3
                                            )

        emod = mhe.estimator.mod
        emod_var_value_dict = {"Ca[8.0]": 0.018655,
                                "Tall[8.0,T]": 384.930409,
                                "Tall[8.0,Tj]": 372.580576,
                                "k[8.0]": 1691.919936,
                                "Tjinb[8.0]": 255.0,
                                "Ca[10.0]": 0.018101,
                                "Tall[10.0,T]": 385.913247,
                                "Tall[10.0,Tj]": 373.945006,
                                "k[10.0]": 1798.164105,
                                "Tjinb[10.0]": 260.0,}
        for comp in emod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_emod = cuid.find_component_on(emod)
            assert comp_emod.value == pytest.approx(emod_var_value_dict[comp],
                                                    1e-3)

        estimator_df = estimator_data.estimator_df
        assert len(estimator_df.index) == 3
        assert estimator_df["mod.Ca[*]"][4] == pytest.approx(
                                        emod_var_value_dict["Ca[8.0]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,T]"][4] == pytest.approx(
                                        emod_var_value_dict["Tall[8.0,T]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,Tj]"][4] == pytest.approx(
                                        emod_var_value_dict["Tall[8.0,Tj]"],
                                        1e-3)
        assert estimator_df["mod.Ca[*]"][6] == pytest.approx(
                                        emod_var_value_dict["Ca[10.0]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,T]"][6] == pytest.approx(
                                        emod_var_value_dict["Tall[10.0,T]"],
                                        1e-3)
        assert estimator_df["mod.Tall[*,Tj]"][6] == pytest.approx(
                                        emod_var_value_dict["Tall[10.0,Tj]"],
                                        1e-3)

        sample_point = mhe.estimator.SAMPLEPOINT_SET
        t_in_second_last_sample_point = [t for t in emod.t
                                         if t > sample_point.at(-3)
                                         and t <= sample_point.at(-2)]
        assert all(emod.Tjinb[t].value == 255.0
                   for t in t_in_second_last_sample_point)
        t_in_last_sample_point = [t for t in emod.t
                                  if t > sample_point.at(-2)
                                  and t <= sample_point.last()]
        assert all(emod.Tjinb[t].value == 260.0
                   for t in t_in_last_sample_point)
