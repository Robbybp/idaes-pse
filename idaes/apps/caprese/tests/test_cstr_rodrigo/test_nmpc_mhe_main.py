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
""" Tests for the example of NMPC and MHE for Rodrigo's cstr
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.examples.cstr_rodrigo.nmpc_mhe_main import (
            setup_controller_estimator,
            solve_first_control_estimation_NLP,
            run_iterations,
            )
from idaes.apps.caprese.dynamic_builder import (
            DynamicSim
            )
from idaes.apps.caprese.data_manager import (
        PlantDataManager,
        ControllerDataManager,
        EstimatorDataManager
        )

class Test_NMPC_MHE_Main(object):

    @pytest.mark.unit
    def test_setup_controller_estimator(self):
        dyna, plant_data, controller_data, estimator_data = \
            setup_controller_estimator(
                                    nmpc_horizon=10,
                                    nmpc_ntfe=5,
                                    nmpc_ntcp=2,
                                    mhe_horizon=10,
                                    mhe_ntfe=10,
                                    mhe_ntcp=2,
                                    plant_ntfe=2,
                                    plant_ntcp=2,
                                    sample_time=2.0
                                        )

        pmod = dyna.plant.mod
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

        cmod = dyna.controller.mod
        var_value_dict = {"Ca[*]": 0.0192,
                           "k[*]": 1818.53008,
                           "Tjinb[*]": 250.0,
                           "Cadot[*]": 0.0,
                           "Tall[*,T]": 384.0,
                           "Tall[*,Tj]": 371.0,
                           "Talldot[*,T]": 0.0,
                           "Talldot[*,Tj]": 0.0,
                           }
        for varstr_slice in var_value_dict.keys():
            cuid = pyo.ComponentUID(varstr_slice)
            for comp in cuid.find_component_on(cmod).values():
                assert comp.value == pytest.approx(
                    var_value_dict[varstr_slice],
                    1e-3,
                    )

        assert hasattr(dyna.controller, 'tracking_objective')
        assert hasattr(dyna.controller, 'pwc_constraint')

        emod = dyna.estimator.mod
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

        actmea_block = dyna.estimator.ACTUALMEASUREMENT_BLOCK
        assert all(mea_comp.value == pytest.approx(var_value_dict["Tall[*,T]"],
                                                   1e-3,)
                   for mea_comp in actmea_block[0].var[:])
        assert all(mea_comp.value == pytest.approx(var_value_dict["Ca[*]"],
                                                   1e-3,)
                   for mea_comp in actmea_block[1].var[:])

        assert hasattr(dyna.estimator, 'noise_minimize_objective')

        assert type(dyna) is DynamicSim
        assert type(plant_data) is PlantDataManager
        assert type(controller_data) is ControllerDataManager
        assert type(estimator_data) is EstimatorDataManager

    @pytest.mark.component
    def test_solve_first_control_estimation_NLP(self):
        dyna, plant_data, controller_data, estimator_data = \
            setup_controller_estimator(
                                    nmpc_horizon=10,
                                    nmpc_ntfe=5,
                                    nmpc_ntcp=2,
                                    mhe_horizon=10,
                                    mhe_ntfe=10,
                                    mhe_ntcp=2,
                                    plant_ntfe=2,
                                    plant_ntcp=2,
                                    sample_time=2.0
                                        )
        dyna, plant_data, controller_data, estimator_data = \
            solve_first_control_estimation_NLP(
                                    dyna,
                                    plant_data,
                                    controller_data,
                                    estimator_data
                                        )

        cmod = dyna.controller.mod
        assert cmod.Tjinb[0].is_fixed()
        assert all(not cmod.Tjinb[i].is_fixed()
                   for i in cmod.Tjinb.index_set()
                   if i>0)

        cmod_var_value_dict = {"Ca[2.0]": 0.01803,
                               "Tall[2.0,T]": 386.04744,
                               "Tall[2.0,Tj]": 374.15439,
                               "k[2.0]":1813.13545,
                               "Tjinb[2.0]": 260.58190,
                               }
        for comp in cmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_cmod = cuid.find_component_on(cmod)
            assert comp_cmod.value == pytest.approx(cmod_var_value_dict[comp],
                                                    1e-3)

        controller_df = controller_data.controller_df
        assert len(controller_df.index) == 2
        assert controller_df["mod.Tjinb[*]"][2] == pytest.approx(
                                        cmod_var_value_dict["Tjinb[2.0]"],
                                        1e-3)

        pmod = dyna.plant.mod
        pmod_var_value_dict = {"Ca[2.0]": 0.01807,
                               "Tall[2.0,T]": 385.96637,
                               "Tall[2.0,Tj]": 374.04397,
                               "k[2.0]":1804.07694,
                               "Tjinb[2.0]": 260.58190,
                               }
        for comp in pmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_pmod = cuid.find_component_on(pmod)
            assert comp_pmod.value == pytest.approx(pmod_var_value_dict[comp],
                                                    1e-3)

        plant_df = plant_data.plant_df
        assert len(plant_df.index) == 5
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

        actmea_block = dyna.estimator.ACTUALMEASUREMENT_BLOCK
        sample_point = dyna.estimator.SAMPLEPOINT_SET
        assert actmea_block[0].var[sample_point.last()].value == pytest.approx(
                                            pmod_var_value_dict["Tall[2.0,T]"],
                                            1e-3
                                            )
        assert actmea_block[1].var[sample_point.last()].value == pytest.approx(
                                            pmod_var_value_dict["Ca[2.0]"],
                                            1e-3
                                            )

        emod = dyna.estimator.mod
        check_control_t_list = [t for t in emod.t
                                if t > sample_point.at(-2)
                                and t <= sample_point.last()]
        assert all(emod.Tjinb[t].is_fixed() for t in check_control_t_list)
        assert all(emod.Tjinb[t].value  == pytest.approx(
                                            pmod_var_value_dict["Tjinb[2.0]"],
                                            1e-3
                                            )
                   for t in check_control_t_list)

        emod_var_value_dict = {"Ca[10]": 0.01809,
                               "Tall[10,T]": 385.96637,
                               "Tall[10,Tj]": 374.03417,
                               "k[10]": 1804.07684,
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
        dyna, plant_data, controller_data, estimator_data = \
            setup_controller_estimator(
                                    nmpc_horizon=10,
                                    nmpc_ntfe=5,
                                    nmpc_ntcp=2,
                                    mhe_horizon=10,
                                    mhe_ntfe=10,
                                    mhe_ntcp=2,
                                    plant_ntfe=2,
                                    plant_ntcp=2,
                                    sample_time=2.0
                                        )
        dyna, plant_data, controller_data, estimator_data = \
            solve_first_control_estimation_NLP(
                                    dyna,
                                    plant_data,
                                    controller_data,
                                    estimator_data
                                        )
        dyna, plant_data, controller_data, estimator_data = run_iterations(
                                                            dyna,
                                                            plant_data,
                                                            controller_data,
                                                            estimator_data,
                                                            iterations=2,
                                                            noise_info=None,
                                                            plot_results=False,
                                                            )
        cmod = dyna.controller.mod
        cmod_var_value_dict = {"Ca[2.0]": 0.01800,
                               "Tall[2.0,T]": 386.09540,
                               "Tall[2.0,Tj]": 374.16755,
                               "k[2.0]": 1818.51318,
                               "Tjinb[2.0]": 260.56909,
                               }
        for comp in cmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_cmod = cuid.find_component_on(cmod)
            assert comp_cmod.value == pytest.approx(cmod_var_value_dict[comp],
                                                    1e-5)

        controller_df = controller_data.controller_df
        assert len(controller_df.index) == 6
        assert controller_df["mod.Tjinb[*]"][6] == pytest.approx(
                                        cmod_var_value_dict["Tjinb[2.0]"],
                                        1e-5)

        pmod = dyna.plant.mod
        pmod_var_value_dict = {"Ca[0.0]": 0.01800,
                                "Tall[0.0,T]": 386.08800,
                                "Tall[0.0,Tj]": 374.16063,
                                "k[0.0]": 1817.68250,
                                "Tjinb[0.0]": 260.56909,
                                "Ca[2.0]": 0.01800,
                                "Tall[2.0,T]": 386.09511,
                                "Tall[2.0,Tj]": 374.16728,
                                "k[2.0]": 1818.48075,
                                "Tjinb[2.0]": 260.56909,
                                }
        for comp in pmod_var_value_dict.keys():
            cuid = pyo.ComponentUID(comp)
            comp_pmod = cuid.find_component_on(pmod)
            assert comp_pmod.value == pytest.approx(pmod_var_value_dict[comp],
                                                    1e-3)

        plant_df = plant_data.plant_df
        assert len(plant_df.index) == 13
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

        actmea_block = dyna.estimator.ACTUALMEASUREMENT_BLOCK
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

        emod = dyna.estimator.mod
        emod_var_value_dict = {"Ca[8.0]": 0.01800,
                               "Tall[8.0,T]": 386.08800,
                               "Tall[8.0,Tj]": 374.16061,
                               "k[8.0]": 1817.68250,
                               "Tjinb[8.0]": 260.57007,
                               "Ca[10.0]": 0.01800,
                               "Tall[10.0,T]": 386.09511,
                               "Tall[10.0,Tj]": 374.16728,
                               "k[10.0]": 1818.48075,
                               "Tjinb[10.0]": 260.56909,}
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

        sample_point = dyna.estimator.SAMPLEPOINT_SET
        t_in_second_last_sample_point = [t for t in emod.t
                                         if t > sample_point.at(-3)
                                         and t <= sample_point.at(-2)]
        assert all(emod.Tjinb[t].value == pytest.approx(260.57007, 1e-3)
                   for t in t_in_second_last_sample_point)
        t_in_last_sample_point = [t for t in emod.t
                                  if t > sample_point.at(-2)
                                  and t <= sample_point.last()]
        assert all(emod.Tjinb[t].value == pytest.approx(260.56909, 1e-3)
                   for t in t_in_last_sample_point)
