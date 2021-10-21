##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Example for Caprese's module for simulation of a plant.
"""
import random
from idaes.apps.caprese.dynamic_builder import DynamicSim
from idaes.apps.caprese.util import apply_noise_with_bounds
from pyomo.environ import SolverFactory, Reference, ComponentUID
from pyomo.dae.initialization import solve_consistent_initial_conditions
# import idaes.logger as idaeslog
from idaes.apps.caprese.examples.cstr_enzyme.cstr_model import make_model
from idaes.apps.caprese.data_manager import PlantDataManager
from idaes.apps.caprese.plotlibrary import (
        plot_plant_state_evolution,
        plot_control_input)
import numpy as np

__author__ = "Kuan-Han Lin"


# See if ipopt is available and set up solver
if SolverFactory('ipopt').available():
    solver = SolverFactory('ipopt')
    solver.options = {
            'tol': 1e-6,
            'bound_push': 1e-8,
            'halt_on_ampl_error': 'yes',
            'linear_solver': 'ma57',
            }
else:
    solver = None


def simulate_model(model, input_data, sample_time):
    m_plant = model
    input_array, inputs = input_data
    time = model.fs.time
    t0 = time.first()

    n_samples = input_array.shape[0]

    # We must identify for the plant which variables are our
    # inputs and measurements.
    input_vardata = [m_plant.find_component(name)[t0] for name in inputs]
    measurements = [
        m_plant.fs.cstr.outlet.conc_mol[0, 'C'],
        m_plant.fs.cstr.outlet.conc_mol[0, 'E'],
        m_plant.fs.cstr.outlet.conc_mol[0, 'S'],
        m_plant.fs.cstr.outlet.conc_mol[0, 'P'],
        m_plant.fs.cstr.outlet.temperature[0],
        m_plant.fs.cstr.volume[0],
    ]

    # Construct the "plant simulator" object
    simulator = DynamicSim(
                    plant_model=m_plant,
                    plant_time_set=m_plant.fs.time,
                    inputs_at_t0=input_vardata,
                    measurements_at_t0=measurements,
                    sample_time=sample_time,
                    )

    plant = simulator.plant

    p_t0 = simulator.plant.time.first()
    p_ts = simulator.plant.sample_points[1]

    # Set up data manager to save plant data
    data_manager = PlantDataManager(plant)
    #--------------------------------------------------------------------------
    solve_consistent_initial_conditions(plant, plant.time, solver)

    data_manager.save_initial_plant_data()

    cinput = input_array[0, :]
    plant.inject_inputs(cinput)

    # This "initialization" really simulates the plant with the new inputs.
    simulator.plant.initialize_by_solving_elements(solver)
    simulator.plant.vectors.input[...].fix() #Fix the input to solve the plant
    solver.solve(simulator.plant, tee = True)
    data_manager.save_plant_data(iteration = 0)

    for i in range(1, n_samples):
        print('\nENTERING SIMULATOR LOOP ITERATION %s\n' % i)

        simulator.plant.advance_one_sample()
        simulator.plant.initialize_to_initial_conditions()
        cinput = input_array[i, :]
        simulator.plant.inject_inputs(cinput)

        simulator.plant.initialize_by_solving_elements(solver)
        simulator.plant.vectors.input[...].fix() #Fix the input to solve the plant
        solver.solve(simulator.plant, tee = True)
        data_manager.save_plant_data(iteration = i)

    return simulator, data_manager


def plot_states_and_inputs(data_manager):
    states_of_interest = data_manager.plant_states_of_interest
    plot_plant_state_evolution(states_of_interest, data_manager.plant_df)
    inputs_to_plot = data_manager.inputs
    plot_control_input(inputs_to_plot, data_manager.plant_df)


def main():
    sample_time = 0.5
    model = make_model(horizon=sample_time, ntfe=5, ntcp=2)

    n_samples = 10
    sample_points = [sample_time*(i+1) for i in range(n_samples)]
    inputs = [
        str(ComponentUID(model.fs.mixer.S_inlet.flow_vol[:])),
        str(ComponentUID(model.fs.mixer.E_inlet.flow_vol[:])),
    ]
    input_values = [2.1, 0.1]
    n_inputs = len(inputs)
    input_array = np.fromiter(
        (input_values[i] for t in sample_points for i in range(n_inputs)),
        float,
    )
    input_array = input_array.reshape((n_samples, n_inputs))
    input_data = (input_array, inputs)

    simulator, data_manager = simulate_model(model, input_data, sample_time)
    plot_states_and_inputs(data_manager)


if __name__ == '__main__':
    main()
