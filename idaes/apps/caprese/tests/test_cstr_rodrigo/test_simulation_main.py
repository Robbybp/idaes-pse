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
""" Tests for the example of plant simulation for Rodrigo's cstr
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.examples.cstr_rodrigo.simulation_main import (
            setup_plant_simulator,
            solve_first_simulation_NLP,
            run_iterations,
            )
from idaes.apps.caprese.dynamic_builder import (
            DynamicSim
            )
from idaes.apps.caprese.data_manager import(
            PlantDataManager,
            EstimatorDataManager,
            )

class TestSimulationMain(object):

    @pytest.mark.unit
    def test_setup_plant_simulator(self):

        simulator, data_manager = setup_plant_simulator(plant_ntfe=4,
                                                        plant_ntcp=2,
                                                        sample_time=2.0)

        pmod = simulator.plant.mod
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

        assert type(simulator) is DynamicSim
        assert type(data_manager) is PlantDataManager

    @pytest.mark.component
    def test_solve_first_simulation_NLP(self):

        simulator, data_manager = setup_plant_simulator(plant_ntfe=4,
                                                        plant_ntcp=2,
                                                        sample_time=2.0)

        simulator, data_manager = solve_first_simulation_NLP(simulator,
                                                             data_manager)

        pmod = simulator.plant.mod
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

        plant_df = data_manager.plant_df
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

    @pytest.mark.component
    def test_run_iterations(self):
        simulator, data_manager = setup_plant_simulator(plant_ntfe=4,
                                                        plant_ntcp=2,
                                                        sample_time=2.0)

        simulator, data_manager = solve_first_simulation_NLP(simulator,
                                                             data_manager)

        simulator, data_manager = run_iterations(simulator,
                                                 data_manager,
                                                 iterations=2,
                                                 plot_results=False)

        pmod = simulator.plant.mod
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

        plant_df = data_manager.plant_df
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
