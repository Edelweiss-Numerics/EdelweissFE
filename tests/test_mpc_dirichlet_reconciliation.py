import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase


class _DummySolver(NonlinearSolverBase):
    def __init__(self):
        self.journal = Journal()
        self.identification = "DummySolver"
        self.options = {}

    def _constrainedDofsOf(self, dirichlet):
        return dirichlet.dofIndices

    def solveIncrement(self, *args, **kwargs):
        pass

    def solveStep(self, *args, **kwargs):
        pass


class _DummyDirichlet:
    def __init__(self, name, dof_indices, delta_values, active=True):
        self.name = name
        self.active = active
        self.dofIndices = np.asarray(dof_indices, dtype=np.int64)
        self.delta = np.asarray(delta_values, dtype=np.float64)

    def reconcileIfSetChanged(self):
        pass


def test_inactive_dirichlet_not_treated_as_prescribed():
    solver = _DummySolver()
    active_bc = _DummyDirichlet("active_bc", [0, 1], [0.0, 0.0], active=True)
    inactive_bc = _DummyDirichlet("inactive_bc", [2, 3], [1.0, 1.0], active=False)

    step_actions = {"dirichlet": {"bc1": active_bc, "bc2": inactive_bc}}

    prescribed = solver._prescribedDofValues(step_actions)
    assert 0 in prescribed
    assert 1 in prescribed
    assert 2 not in prescribed
    assert 3 not in prescribed


def test_reconciliation_redundant_and_weight_normalization():
    solver = _DummySolver()
    # BC prescribes slave DOF 10 = 0.0 and master DOF 20 = 0.0, but NOT master DOF 30
    bc = _DummyDirichlet("bc", [10, 20], [0.0, 0.0], active=True)
    step_actions = {"dirichlet": {"bc": bc}}

    # Record 1: slave 10 = 0.5 * master 20 + 0.5 * master 30 (master 30 unprescribed, total weight 1.0)
    # Record 2: slave 11 (unprescribed slave) = master 20
    records = [
        (10, [(20, 0.5), (30, 0.5)]),
        (11, [(20, 1.0)]),
    ]

    kept = solver._reconcileMPCDirichletConflicts(records, step_actions)
    # Slave 10 is prescribed, so its constraint record should be dropped in favour of BC
    # Slave 11 is not prescribed, so kept
    assert len(kept) == 1
    assert kept[0][0] == 11
