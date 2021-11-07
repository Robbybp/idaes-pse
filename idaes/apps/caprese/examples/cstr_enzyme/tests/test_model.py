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
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.dae as dae

from pyomo.common.collections import ComponentSet
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.incidence_analysis import (
    IncidenceGraphInterface,
    solve_strongly_connected_components,
)
from pyomo.contrib.incidence_analysis.interface import (
    _generate_variables_in_constraints,
)

from idaes.apps.caprese.examples.cstr_enzyme.cstr_model import make_model

import pytest


DAE_DISC_SUFFIX = "_disc_eq"
INPUTS = [
    "fs.mixer.S_inlet_state[0].flow_vol",
    "fs.mixer.S_inlet_state[0].conc_mol[C]",
    "fs.mixer.S_inlet_state[0].conc_mol[E]",
    "fs.mixer.S_inlet_state[0].conc_mol[S]",
    "fs.mixer.S_inlet_state[0].conc_mol[P]",
    "fs.mixer.S_inlet_state[0].conc_mol[Solvent]",
    "fs.mixer.S_inlet_state[0].temperature",
    "fs.mixer.E_inlet_state[0].flow_vol",
    "fs.mixer.E_inlet_state[0].conc_mol[C]",
    "fs.mixer.E_inlet_state[0].conc_mol[E]",
    "fs.mixer.E_inlet_state[0].conc_mol[S]",
    "fs.mixer.E_inlet_state[0].conc_mol[P]",
    "fs.mixer.E_inlet_state[0].conc_mol[Solvent]",
    "fs.mixer.E_inlet_state[0].temperature",
]
OUTPUTS = [
    "fs.cstr.control_volume.properties_out[0].flow_vol",
    "fs.cstr.control_volume.properties_out[0].conc_mol[C]",
    "fs.cstr.control_volume.properties_out[0].conc_mol[E]",
    "fs.cstr.control_volume.properties_out[0].conc_mol[S]",
    "fs.cstr.control_volume.properties_out[0].conc_mol[P]",
    "fs.cstr.control_volume.properties_out[0].conc_mol[Solvent]",
    "fs.cstr.control_volume.properties_out[0].temperature",
]


def _generate_deriv_diff_disc_wrt(m, s):
    # TODO: This function should likely be promoted elsewhere
    for deriv in m.component_objects(pyo.Var):
        if isinstance(deriv, dae.DerivativeVar):
            if s in ComponentSet(deriv.get_continuousset_list()):
                state = deriv.get_state_var()
                name = deriv.local_name
                block = deriv.parent_block()
                disc = block.find_component(name + DAE_DISC_SUFFIX)
                assert disc is not None
                deriv_dict = dict(slice_component_along_sets(deriv, (s,)))
                state_dict = dict(slice_component_along_sets(deriv, (s,)))
                disc_dict = dict(slice_component_along_sets(deriv, (s,)))
                for idx in deriv_dict:
                    yield deriv_dict[idx], state_dict[idx], disc_dict[idx]


@pytest.mark.unit
class TestEnzymeCSTRModelVariables(unittest.TestCase):

    def test_inputs(self):
        # A simple test to make sure we have the inputs we expect
        inputs = INPUTS
        m = make_model(steady=True)
        input_vars = [m.find_component(name) for name in inputs]
        self.assertTrue(all(var is not None for var in input_vars))

    def test_differential(self):
        # Make sure we don't have any differential variables
        m = make_model(steady=True)
        time = m.fs.time
        deriv_diff_disc = list(_generate_deriv_diff_disc_wrt(m, time))
        self.assertEqual(len(deriv_diff_disc), 0)

    def test_outputs(self):
        outputs = OUTPUTS
        m = make_model(steady=True)
        output_vars = [m.find_component(name) for name in outputs]
        self.assertTrue(all(var is not None for var in output_vars))


@pytest.mark.component
class TestEnzymeCSTRModelSteadyState(unittest.TestCase):

    def test_default_struct_nonsingular(self):
        m = make_model(steady=True)
        m._obj = pyo.Objective(expr=0.0)
        nlp = PyomoNLP(m)
        jac = nlp.evaluate_jacobian_eq()
        constraints = list(m.component_data_objects(pyo.Constraint, active=True))
        variables = list(_generate_variables_in_constraints(constraints))
        self.assertEqual(len(variables), len(constraints))
        igraph = IncidenceGraphInterface()
        matching = igraph.maximum_matching(variables, constraints)
        self.assertEqual(len(matching), len(constraints))

    def test_default_steady_state(self):
        m = make_model(steady=True)

        m3min = pyo.units.m**3/pyo.units.min
        kmolm3 = pyo.units.kmol/pyo.units.m**3
        K = pyo.units.K
        input_values = {
            "fs.mixer.S_inlet_state[0].flow_vol": 2.1*m3min,
            "fs.mixer.S_inlet_state[0].conc_mol[C]": 0.0*kmolm3,
            "fs.mixer.S_inlet_state[0].conc_mol[E]": 0.0*kmolm3,
            "fs.mixer.S_inlet_state[0].conc_mol[S]": 12.92*kmolm3,
            "fs.mixer.S_inlet_state[0].conc_mol[P]": 0.0*kmolm3,
            "fs.mixer.S_inlet_state[0].conc_mol[Solvent]": 1.0,
            "fs.mixer.S_inlet_state[0].temperature": 329.24*K,
            "fs.mixer.E_inlet_state[0].flow_vol": 0.1*m3min,
            "fs.mixer.E_inlet_state[0].conc_mol[C]": 0.0*kmolm3,
            "fs.mixer.E_inlet_state[0].conc_mol[E]": 11.91*kmolm3,
            "fs.mixer.E_inlet_state[0].conc_mol[S]": 0.0*kmolm3,
            "fs.mixer.E_inlet_state[0].conc_mol[P]": 0.0*kmolm3,
            "fs.mixer.E_inlet_state[0].conc_mol[Solvent]": 1.0,
            "fs.mixer.E_inlet_state[0].temperature": 310.0*K,
        }
        self.assertEqual(set(input_values), set(INPUTS))
        for name, val in input_values.items():
            # Here we assert that the above inputs are valid in our
            # model, and that these are their default values
            var = m.find_component(name)
            self.assertFalse(var is None)
            self.assertEqual(var.value, pyo.value(val))
            # This check fails as my variables don't have units
            #self.assertEqual(
            #    var.get_units().to_string(),
            #    pyo.units.get_units(val).to_string()
            #)

        # These default conditions, with default initialization, are a great
        # example of a solve that converges when decomposed by SCC, but
        # not otherwise. TODO: debug why this problem does not converge
        # when attempted simultaneously.
        solver = pyo.SolverFactory("ipopt")
        solver.options["halt_on_ampl_error"] = "yes"
        solve_strongly_connected_components(m, solver)
        solver.solve(m)

        with open("_newsol", "w") as fp:
            for var in m.component_data_objects(pyo.Var):
                fp.write(
                    var.name
                    + " %s"%(
                        var.value * var.get_units() 
                        if var.get_units() is not None
                        else var.value
                    )
                    + "\n"
                )


if __name__ == "__main__":
    #unittest.main()
    TestEnzymeCSTRModelSteadyState().test_default_steady_state()
