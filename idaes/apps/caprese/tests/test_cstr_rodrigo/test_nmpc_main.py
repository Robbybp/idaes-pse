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
""" Tests for the example of NMPC for Rodrigo's cstr
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.examples.cstr_rodrigo.nmpc_main import (
            setup_controller,
            solve_first_control_NLP,
            run_iterations,
            )
from idaes.apps.caprese.dynamic_builder import (
            DynamicSim
            )
from idaes.apps.caprese.data_manager import(
            PlantDataManager,
            ControllerDataManager,
            )

class TestNMPCMain(object):

    @pytest.mark.unit
    def test_setup_controller(self):
        nmpc, plant_data, controller_data = setup_controller(nmpc_horizon=10,
                                                             nmpc_ntfe=5,
                                                             nmpc_ntcp=2,
                                                             plant_ntfe=2,
                                                             plant_ntcp=2,
                                                             sample_time=2.0)

        pmod = nmpc.plant.mod
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

        cmod = nmpc.controller.mod
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

        assert hasattr(nmpc.controller, 'tracking_objective')
        assert hasattr(nmpc.controller, 'pwc_constraint')

        assert type(nmpc) is DynamicSim
        assert type(plant_data) is PlantDataManager
        assert type(controller_data) is ControllerDataManager

    @pytest.mark.component
    def test_solve_first_control_NLP(self):
        nmpc, plant_data, controller_data = setup_controller(
                                            nmpc_horizon=10,
                                            nmpc_ntfe=5,
                                            nmpc_ntcp=2,
                                            plant_ntfe=2,
                                            plant_ntcp=2,
                                            sample_time=2.0
                                            )

        nmpc, plant_data, controller_data = solve_first_control_NLP(
                                            nmpc,
                                            plant_data,
                                            controller_data
                                            )

        cmod = nmpc.controller.mod
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

        pmod = nmpc.plant.mod
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

    @pytest.mark.component
    def test_run_iterations(self):
        nmpc, plant_data, controller_data = setup_controller(
                                            nmpc_horizon=10,
                                            nmpc_ntfe=5,
                                            nmpc_ntcp=2,
                                            plant_ntfe=2,
                                            plant_ntcp=2,
                                            sample_time=2.0
                                            )

        nmpc, plant_data, controller_data = solve_first_control_NLP(
                                            nmpc,
                                            plant_data,
                                            controller_data
                                            )

        nmpc, plant_data, controller_data = run_iterations(
                                            nmpc,
                                            plant_data,
                                            controller_data,
                                            iterations = 2,
                                            noise_info = None,
                                            plot_results = False)

        cmod = nmpc.controller.mod
        cmod_var_value_dict = {"Ca[2.0]": 0.01800,
                               "Tall[2.0,T]": 386.09540,
                               "Tall[2.0,Tj]": 374.16755,
                               "k[2.0]": 1818.51323,
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

        pmod = nmpc.plant.mod
        pmod_var_value_dict = {"Ca[2.0]": 0.01800,
                                "Tall[2.0,T]": 386.09511,
                                "Tall[2.0,Tj]": 374.16727,
                                "k[2.0]":1818.48086,
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
