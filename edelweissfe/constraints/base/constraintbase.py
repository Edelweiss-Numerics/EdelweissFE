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
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.checkpointedstate import packState, unpackState
from edelweissfe.utils.schema import OptionSchemaProvider
from edelweissfe.variables.scalarvariable import ScalarVariable


class ConstraintBase(OptionSchemaProvider, ABC, VIJEntityBase):
    #: Scratch buffer for the tangent that :meth:`computeKernelsExplicit` computes and discards;
    #: re-allocated only when the DOF footprint changes.
    _discardedTangentScratch = None

    #: The state this constraint carries from one increment to the next, by attribute name and type;
    #: see :mod:`~edelweissfe.utils.checkpointedstate`. Every constraint declares it (empty if it carries
    #: none) or overrides :meth:`getRestartData` and :meth:`setRestartData`.
    checkpointedState: dict | None = None

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "ConstraintBase":
        """Create this constraint from a parsed ``.inp`` constraint definition.

        The one place a module's input-file shape (set/surface *names*, string-typed booleans) is
        turned into the typed arguments its real constructor takes. Override it together with a
        typed ``__init__``; leave it alone and the dict-consuming constructor is used unchanged.

        Parameters
        ----------
        name
            The name of the constraint.
        definition
            The parsed option mapping for this constraint (the datalines-derived ``kwargs``).
        model
            The model tree.

        Returns
        -------
        ConstraintBase
            The constructed constraint.
        """
        return cls(name, model, **definition)

    @abstractmethod
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        """The constraint base class.

        Constraints can act on nodal variables, and scalar variables.
        If scalar variables are required, the can be created on demand by
        defining
        :func:`~ConstraintBase.getNumberOfAdditionalNeededScalarVariables` and
        :func:`~ConstraintBase.assignAdditionalScalarVariables`, which are called at the beginning of an analysis.

        If scalar variables are used, EdelweissFE expects the layout of the residual contribution ``P``
        (and of the tangent) to be of the form

        .. code-block:: console

            [ node 1 - dofs field 1,
              node 1 - dofs field 2,
              node 1 - ... ,
              node 1 - dofs field n,
              node 2 - dofs field 1,
              ... ,
              node N - dofs field n,
              scalar variable 1,
              scalar variable 2,
              ... ,
              scalar variable J ].

        Parameters
        ----------
        name
            The name of the constraint.
        model
            A dictionary containing the model tree.
        kwargs
            Key value pairs.
        """

        self.scalarVariables = []

    @property
    @abstractmethod
    def nodes(self) -> list:
        """The nodes this constraint is acting on."""

    @property
    @abstractmethod
    def fieldsOnNodes(self) -> list:
        """The fields on the nodes this constraint is acting on."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with."""

    def updateConnectivity(self, model: FEModel) -> bool:
        """Called once at the start of every increment, before the global equation system is
        (re)created, so a constraint can refresh a dynamic candidate set (e.g. proximity-based
        contact pairs) and declare whether its ``nodes``/``fieldsOnNodes``/``nDof`` footprint has
        changed since the last call.

        The default implementation does nothing and reports no change, which is correct for every
        constraint whose DOF footprint is fixed at construction (i.e. every constraint that does
        not override this method).

        Parameters
        ----------
        model
            The current model.

        Returns
        -------
        bool
            True if this constraint's contribution to the global system has changed and the
            equation system must be rebuilt before the next Newton solve.
        """

        return False

    def acceptLastState(self):
        """Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` when an increment
        is accepted, so a stateful constraint (e.g. frictional contact) can promote the state of
        the last (converged) Newton iterate to its history.

        The default implementation does nothing, which is correct for every stateless constraint
        (i.e. every constraint that does not override this method)."""

    def getRestartData(self) -> dict[str, np.ndarray]:
        """The state this constraint carries from one increment to the next; by default the attributes
        declared in :attr:`checkpointedState`. Overridden where the state is not a plain attribute.

        Returns
        -------
        dict[str, numpy.ndarray]
            A flat mapping of array name to array.
        """

        return packState(self)

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the state :meth:`getRestartData` returned.

        Parameters
        ----------
        data
            The mapping of arrays.
        """

        unpackState(self, data)

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """This method is called to determine the additional number of scalar variables
        this Constraint needs.

        Returns
        -------
        int
            Number of ScalarVariables required.

        """

        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """Assign a list of scalar variables associated with this constraint.

        Parameters
        ----------
        list
            The list of ScalarVariables associated with this constraint.

        """

        self.scalarVariables = scalarVariables

    def computeKernelsExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add this constraint's contribution to the residual, without a tangent.

        The counterpart of :meth:`computeKernels` for an explicit solver, which solves no linear
        system and so has no use for a tangent. This default calls :meth:`computeKernels` with a
        scratch tangent that is thrown away, so that every constraint also works in an explicit
        analysis. A constraint whose tangent is costly overrides it, e.g. through
        :class:`~edelweissfe.constraints.base.forcesonlyexplicitevaluation.ForcesOnlyExplicitEvaluation`.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The current solution increment, likewise restricted.
        P
            The residual contribution to augment, with the sign described in :meth:`computeKernels`.
        timeStep
            The current time step.
        """

        # The discarded tangent buffer is kept across increments and re-zeroed rather than reallocated.
        size = self.getVIJContributionSize()
        scratch = self._discardedTangentScratch
        if scratch is None or scratch.size != size:
            scratch = self._discardedTangentScratch = np.zeros(size)
        else:
            scratch[:] = 0.0

        K = self.shapeVIJContribution(scratch)
        self.computeKernels(U_np, dU, P, K, timeStep)

    @abstractmethod
    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add this constraint's contribution ``P`` to the residual, and its tangent ``K = dP/dU``.

        A constraint follows the rule every element and every load follows (see
        :ref:`residual-and-tangent-convention`): ``P`` is the force the constraint needs at its
        degrees of freedom, i.e. the *negative* of the force it exerts on the nodes, exactly as an
        element of the same mechanics would add its internal force. A penalty spring of stiffness
        :math:`k` that pulls a degree of freedom :math:`u` towards :math:`\\bar{u}`, for example, adds

        .. math::

            P \\mathrel{+}= k\\,(u - \\bar{u}), \\qquad K \\mathrel{+}= k .

        A Lagrange multiplier constraint :math:`g(\\boldsymbol{u}) = 0` adds
        :math:`\\boldsymbol{\\lambda}^T \\partial g / \\partial \\boldsymbol{u}` to the rows of its nodes
        and :math:`g` to the rows of its multipliers.

        Both ``P`` and ``K`` are augmented, never overwritten. ``K`` must be the derivative of what is
        added to ``P``: a test can then check it by finite differences, which also catches a ``P``
        of the wrong sign.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom; its own scalar
            variables (e.g. Lagrange multipliers) come last.
        dU
            The solution increment since the last converged state, likewise restricted.
        P
            The residual contribution to augment.
        K
            The tangent ``dP/dU`` to augment, in the shape :meth:`shapeVIJContribution` gives it (a
            dense ``nDof x nDof`` view unless the constraint declares a sparser pattern). A
            constraint that inherits
            :class:`~edelweissfe.constraints.base.forcesonlyexplicitevaluation.ForcesOnlyExplicitEvaluation`
            is also called with ``K=None`` and must then skip the tangent.
        timeStep
            The current time step.
        """
