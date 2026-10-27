#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""p-multigrid (Galerkin P1 corner-node) preconditioner for one field's diagonal block: the cycle of
``TwoLevel(coarseSpace=P1Interpolation(coarse))`` (:class:`~edelweissfe.linsolve.composition.twolevel.TwoLevel`).

Precondition the quadratic operator through its linear (P1) part: :math:`\\nu` Chebyshev sweeps on the field's own
(scaled) block, restrict the residual through :math:`P^T`, solve on the Galerkin-projected :math:`A_c = P^T A P` with
the coarse component, prolong through :math:`P`, :math:`\\nu` more Chebyshev sweeps.

The prolongation interpolates the scaled variables
---------------------------------------------------

The block is :math:`\\hat A = D A D` (a :class:`~edelweissfe.linsolve.composition.diagonalscaling.DiagonalScaling`
above), and a displacement :math:`u` of the physical problem is :math:`\\hat u = D^{-1} u` in its variables. The P1
interpolant acts on :math:`u`: :math:`P_{geo}` keeps a corner node's displacement and gives a midside node the mean of
its two edge-endpoint corners. In the scaled variables the same interpolation is

.. math::

    P = D^{-1} P_{geo} D_c: \\quad P_{cc} = 1, \\qquad
    P_{ma} = \\tfrac{1}{2} \\frac{s_a}{s_m}, \\quad P_{mb} = \\tfrac{1}{2} \\frac{s_b}{s_m}

for a midside dof :math:`m` on the edge :math:`(a, b)`, with :math:`s` the diagonal of :math:`D` and :math:`D_c` its
restriction to the corners. The plain weights :math:`\\tfrac{1}{2}, \\tfrac{1}{2}` would interpolate :math:`\\hat u`
as if it were :math:`u`: a scaled translation :math:`D^{-1} t`, the smoothest error there is, would be off by the
ratios :math:`s_a / s_m` at every midside node (12 % in the median, up to a factor of 2000 measured on one model),
and the coarse level could not remove it. With :math:`P` the translations are reproduced exactly, and

.. math::

    A_c = P^T \\hat A P = D_c \\, P_{geo}^T A P_{geo} \\, D_c

is the P1 operator of the physical problem in the corners' scaled variables: the operator of the corner-node field,
scaled by :math:`D_c`. Without a scaling above (:math:`D = I`), :math:`P = P_{geo}`.

The coarse problem is a field of its own
----------------------------------------

The coarse solve is a component of the composed solver, set up on :math:`A_c` with the context of the corner-node field
(:meth:`~edelweissfe.linsolve.composition.base.SetUpContext.ofCoarseNodes`: the same field on the coarse nodes, their
coordinates, the scaling :math:`D_c`), so that every component works on it unchanged. Its default,
``DiagonalScaling(TwoLevel(coarseSpace=RigidBodyModes()))``, scales :math:`A_c` to a unit diagonal once more: where a
midside diagonal is tiny against its corners', the weights :math:`s_a / s_m` reach about 1000,
:math:`A_c` inherits entries of that size squared, and the spectral radius the coarse level's own Chebyshev smoother
works with was measured at :math:`1.8 \\cdot 10^5` instead of about 5. Measurements:
:ref:`measured-scaled-p1`.

The cycle runs in C++ (``PTwoGridPreconditionerT`` in ``edelweissfe/linsolve/amgcl/amgcl-wrapper.hpp``, wrapped as
:class:`~edelweissfe.linsolve.amgcl.amgcl.PyAMGCLPTwoGrid`), OpenMP-threaded throughout, and calls the coarse
component back once per application.

**Dirichlet handling is load-bearing, do not simplify away.** A field's diagonal block still carries
its Dirichlet identity rows (production applies Dirichlet elimination upstream of the linear solver), and
the identity rows of multi-point-constraint slaves, which are treated alike. Both
the fine smoothing and the Galerkin projection must act on the *free* DOFs only, with the Dirichlet
DOFs decoupled entirely, not merely masked in place -- masking in place for the fine smoother
alone was tried and diverged outright. A free midside node whose edge-endpoint corner is Dirichlet-constrained keeps
only the surviving weight of its free endpoint -- no renormalization (the constrained corner contributes exactly zero
to a homogeneous Newton correction, which the dropped weight already encodes). A free midside dof whose *both*
endpoint dofs are constrained (a symmetry plane and a tie on one edge, issue #223) would get an empty row of :math:`P`
-- no coarse correction at all -- so its node becomes a coarse node of its own (:attr:`PTwoGridPreconditioner.coarseNodes`),
as :func:`~edelweissfe.numerics.p1topology.buildP1Map` makes a node a corner on an AMR edge conflict; it is decided at the
build, where the Dirichlet rows are known, not in the topology map. A Dirichlet coarse dof is a zero column of :math:`P`
and an identity row of :math:`A_c`: the coarse field keeps its Dirichlet rows, as the fine one does.

The C++ implementation does not extract the free submatrix, it decouples the Dirichlet DOFs exactly
inside the full block layout instead: the fine right-hand side and iterate are kept zero on them (a
Dirichlet row holds only its diagonal, so every residual and every Chebyshev update stays zero
there, whatever the columns hold), and ``P`` has no rows for them. That is the
free-submatrix cycle up to floating-point summation order, and it is what allows a node-wise 3x3
block storage of the fine operator (``fineBlockSize``) on partially constrained nodes.
"""

import numpy as np
import scipy.sparse as sp

from edelweissfe.linsolve.composition.twolevelcycle import TwoLevelPreconditioner
from edelweissfe.utils import performancetiming


class PTwoGridPreconditioner(TwoLevelPreconditioner):
    """A p-two-grid preconditioner for one vector field's diagonal block, a two-level preconditioner
    (:class:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner`) whose coarse level is the P1
    corner nodes, and whose coarse solve is a callable.

    Parameters
    ----------
    isCorner, edgeEndpoints
        This field's P1 topology map (:func:`edelweissfe.numerics.p1topology.buildP1Map`), in the
        field's own node order.
    coarseSolve
        The coarse solve, a callable ``r_c -> z_c`` with :math:`z_c \\approx A_c^{-1} r_c`, e.g. a component of the
        composed solver once it is set up on the coarse operator :meth:`build` returns; called once per application.
    amgclTree, backendBlockSize, backendPrecision
        The parameters of the cycle, see
        :meth:`~edelweissfe.linsolve.composition.twolevelcycle.TwoLevelPreconditioner.cycleParameters`: the fine
        Chebyshev smoother is the ``relax``, ``npre`` and ``npost`` of the field's AMGCL tree. With ``npre`` 0 the cycle
        starts with the coarse-grid correction of the right-hand side itself and ends with ``npost`` sweeps: one
        smoother application fewer per cycle than a symmetric one, at the price of a nonsymmetric preconditioner -- of
        no concern to the right-preconditioned GMRES around it, which needs a fixed linear operator, not a symmetric
        one. Ending on the smoother is the right way round for that one sweep: it damps the high-frequency error the
        prolongated correction leaves on the fine level.
    """

    def __init__(
        self,
        isCorner: np.ndarray,
        edgeEndpoints: np.ndarray,
        coarseSolve,
        amgclTree: dict,
        backendBlockSize: int = 1,
        backendPrecision: str = "double",
    ):
        self._isCorner = np.ascontiguousarray(isCorner, dtype=np.int8)
        self._edgeEndpoints = np.ascontiguousarray(edgeEndpoints, dtype=np.int32)
        from edelweissfe.linsolve.amgcl.amgcl import PyAMGCLPTwoGrid

        super().__init__(
            PyAMGCLPTwoGrid,
            self.cycleParameters(amgclTree, backendBlockSize, backendPrecision),
        )
        self._preconditioner.setCoarseSolve(coarseSolve)

    @classmethod
    def parameterRejection(cls, **parameters) -> "str | None":
        """Why these settings -- keywords of this class -- cannot be used, or ``None``, asked without a topology
        map and a coarse solve."""
        return super().parameterRejection(np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=int), None, **parameters)

    def build(self, A, scaling: np.ndarray) -> sp.csr_matrix:
        """Build the prolongation, the Galerkin coarse operator and the fine Chebyshev smoother, all in C++, and return
        the coarse operator, on which the caller sets up the coarse solve before the first application.

        Parameters
        ----------
        A
            This field's scaled diagonal block :math:`\\hat A = D A D` (scipy CSR), still carrying its Dirichlet
            identity rows.
        scaling
            This field's slice of the scaling :math:`D`: the weights of the prolongation (see the module documentation).

        Returns
        -------
        sp.csr_matrix
            :math:`A_c = P^T \\hat A P` on all dofs of the :attr:`coarseNodes`, in node order and then by component,
            with an identity row for each Dirichlet coarse dof; 0 x 0 for a field without a free dof (the cycle is then the
            exact diagonal inverse and needs no coarse solve).
        """
        A = A.tocsr()
        n = A.shape[0]
        nNodes = len(self._isCorner)
        if (nNodes == 0) != (n == 0) or (nNodes > 0 and n % nNodes != 0):
            raise ValueError(
                "ptwogrid: block size {:} is not a multiple of the topology map's node count {:} -- "
                "the map does not match this field.".format(n, nNodes)
            )
        with performancetiming.timeit("C++ build"):
            # a build that throws leaves the C++ object unbuilt: applyPreconditioner then raises
            return self._preconditioner.build(
                A,
                scaling,
                self._isCorner,
                self._edgeEndpoints,
            )

    @property
    def nCoarse(self) -> int:
        """The coarse dofs: all dofs of the :attr:`coarseNodes`."""
        return self._preconditioner.nCoarse

    @property
    def coarseNodes(self) -> np.ndarray:
        """Per node, whether the last build made it a coarse node: the corner nodes of the P1 map, and every midside node
        with a free dof whose two edge-endpoint dofs are both constrained (see the module documentation)."""
        return self._preconditioner.coarseNodes

    def prolongationMatrix(self) -> sp.csr_matrix:
        """The prolongation ``P`` as the cycle uses it, a scipy CSR matrix (for inspection)."""
        return self._preconditioner.prolongationMatrix()
