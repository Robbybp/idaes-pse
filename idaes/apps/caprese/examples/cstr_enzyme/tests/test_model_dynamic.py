import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.dae.flatten import flatten_dae_components
from pyomo.contrib.incidence_analysis.util import (
    solve_strongly_connected_components,
)

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.apps.caprese.examples.cstr_enzyme.cstr_model import make_model

"""
The primary purpose of this test is to confirm that the dynamic
CSTR model gives approximately the same results as the paper from
which it was taken.

"""


def get_scalar_data_of_time_indexed_variables(model, time):
    t0 = next(iter(time))
    scalar_vars, dae_vars = flatten_dae_components(model, time, pyo.Var)
    data = {
        str(pyo.ComponentUID(var.referent)): var[t0].value 
        for var in dae_vars
    }
    return data


def load_scalar_data_into_model_at_time(model, data, time):
    for name, val in data.items():
        var = model.find_component(name)
        for t in time:
            var[t].set_value(val)


class TestDynamicModel(unittest.TestCase):

    def _get_initial_steady_data(self):
        m = make_model(steady=True)
        time = m.fs.time
        solver = pyo.SolverFactory("ipopt")
        solve_strongly_connected_components(
            m, solver, solve_kwds={"tee": False}
        )
        solver.solve(m, tee=False)
        data = get_scalar_data_of_time_indexed_variables(m, time)
        return data

    def _get_target_steady_data(self):
        m = make_model(steady=True)
        time = m.fs.time
        cv = m.fs.cstr.control_volume
        cv.heat[:].fix(-4300.0/900.0/0.231)
        solver = pyo.SolverFactory("ipopt")
        solve_strongly_connected_components(
            m, solver, solve_kwds={"tee": False}
        )
        solver.solve(m, tee=False)
        data = get_scalar_data_of_time_indexed_variables(m, time)
        return data

    def test_dynamic(self):
        m = make_model(steady=False)
        time = m.fs.time
        cv = m.fs.cstr.control_volume
        self.assertEqual(degrees_of_freedom(m), 0)

        data = self._get_initial_steady_data()
        load_scalar_data_into_model_at_time(m, data, time)
        target_data = self._get_target_steady_data()
        # FIXME:
        # Does the response I'm seeing here make sense?
        # Does my model accurately capture my target operating point?
        # The target operating point seems to be wrong
        # (11.23 instead of 12.0)
        # Why do I expect a target steady state of 12.0?
        # Where does heat appear? In the differential equation for enthalpy
        en_bal = next(iter(cv.enthalpy_balances.values()))
        from pyomo.core.expr.visitor import identify_variables
        for var in identify_variables(en_bal.expr):
            print(var.name, var.get_units())
        import pdb; pdb.set_trace()

        for t in time:
            if t != time.first():
                cv.heat[t].fix(-4300/900.0/0.231)

        solver = pyo.SolverFactory("ipopt")
        solve_strongly_connected_components(
            m, solver, solve_kwds={"tee": False}
        )
        solver.solve(m, tee=True)
        for t in m.fs.time:
            print(cv.properties_out[t].conc_mol["S"].value)


if __name__ == "__main__":
    unittest.main()
