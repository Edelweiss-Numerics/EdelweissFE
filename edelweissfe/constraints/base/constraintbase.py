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

from abc import abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.systemmatrixcontributor import (
    SystemMatrixContributorBase,
    countEntriesOfBlock,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.checkpointedstate import packState, unpackState
from edelweissfe.utils.schema import OptionSchemaProvider
from edelweissfe.variables.scalarvariable import ScalarVariable


class ConstraintBase(OptionSchemaProvider, BaseNodeCouplingEntity, SystemMatrixContributorBase):
    #: Scratch buffer for the tangent that :meth:`computeKernelsExplicit` computes and discards, sized for the block
    #: the constraint has in the discrete system it is evaluated in; None until its first evaluation there (see
    #: :meth:`prepareExplicitEvaluation`).
    _discardedTangentScratch = None

    #: The state this constraint carries from one increment to the next, by attribute name and type;
    #: see :mod:`~edelweissfe.utils.checkpointedstate`. Every constraint declares it (empty if it carries
    #: none) or overrides :meth:`getRestartData` and :meth:`setRestartData`.
    checkpointedState: dict | None = None

    #: Whether this constraint is within reach of what it constrains, as it finds itself in :meth:`updateConnectivity`:
    #: e.g. a contact constraint whose partner is far away is not, and takes no part in the discrete system then (see
    #: :attr:`active`). A constraint sets it there; the model sees the change of :attr:`active` itself (see
    #: :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`). A constraint that does not check stays within
    #: reach.
    withinReach: bool = True

    #: Whether this constraint is switched on from outside; see :attr:`switchedOn`.
    _switchedOn: bool = True

    @property
    def switchedOn(self) -> bool:
        """Whether this constraint is switched on from outside, e.g. by a ``modelupdate`` step action, which switches it
        through :attr:`active`. A constraint switched off from outside rests: the model does not locate it anew
        (:meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`), it accepts no state (:meth:`acceptLastState`),
        so that its history stays as it was, and the solvers neither evaluate it nor refuse it. A solver that cannot
        handle a constraint refuses it while it is switched on, even out of reach: it may come within reach during the
        step.

        Returns
        -------
        bool
            Whether the constraint is switched on.
        """

        return self._switchedOn

    @property
    def active(self) -> bool:
        """Whether this constraint takes part in the discrete system, while it is :attr:`switchedOn` and
        :attr:`withinReach`. A constraint that does not take part gets no degrees of freedom and adds nothing to the
        residual or the tangent; its scalar variables (e.g. Lagrange multipliers) leave the discrete system with it, so
        that they leave no empty rows behind.

        Setting it switches the constraint on or off from outside, e.g. by a ``modelupdate`` step action at the start
        of a step. A switch-off from outside lasts until the constraint is switched on again from outside, whatever the
        constraint finds in reach; switched on again, it takes part while it is within reach.

        A solver evaluates exactly the constraints that took part when its discrete system was built, and the model
        sees a change of ``active`` only through the constraints it updates itself
        (:meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`). Therefore ``active`` is switched from outside
        only between steps, by a ``modelupdate`` before the step begins, which starts from a new discrete system.

        Returns
        -------
        bool
            Whether the constraint takes part in the discrete system.
        """

        return self._switchedOn and self.withinReach

    @active.setter
    def active(self, switchedOn: bool):
        self._switchedOn = switchedOn

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
    def nNodes(self) -> int:
        """The number of nodes this constraint is acting on."""

        return len(self.nodes)

    @property
    @abstractmethod
    def fields(self) -> list:
        """The fields on the nodes this constraint is acting on."""

    @property
    def dofIndicesPermutation(self) -> None:
        """None: the layout of a constraint is nodewise, followed by its scalar variables."""

        return None

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with."""

    def updateConnectivity(self, model: FEModel) -> bool:
        """Called once at the start of every increment, before the global discrete system is
        (re)created, so a constraint can refresh a dynamic candidate set (e.g. proximity-based
        contact pairs) and declare whether the nodes it couples (``nodes``/``fields``/``nDof``) have
        changed since the last call. A constraint that is not always within reach also sets
        :attr:`withinReach` here; it does not report the change of :attr:`active` itself, the model compares it (see
        :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`).

        The default implementation does nothing and reports no change, which is correct for every
        constraint whose nodes are fixed at construction (i.e. every constraint that does
        not override this method).

        Parameters
        ----------
        model
            The current model.

        Returns
        -------
        bool
            True if the nodes this constraint couples have changed.
        """

        return False

    def acceptLastState(self):
        """Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` when an increment
        is accepted, so a stateful constraint (e.g. frictional contact) can promote the state of
        the last (converged) Newton iterate to its history. Only an :attr:`active` constraint is called: one that took
        no part in the increment keeps its history.

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

    def prepareExplicitEvaluation(self):
        """Called by an explicit solver whenever the constraint is located anew in its discrete system (a rebuilt
        one, or one whose constraints are re-located), before it evaluates the constraint there with
        :meth:`computeKernelsExplicit`.

        This default forgets the tangent that :meth:`computeKernelsExplicit` throws away, so that the next evaluation
        sizes it, once, for the block the constraint has now. The size is the number of entries of the block
        (:func:`~edelweissfe.numerics.systemmatrixcontributor.countEntriesOfBlock`), which for a constraint that lists
        its entries means building the list: once per discrete system, not in every increment. A constraint that
        evaluates its forces without a tangent needs nothing here.

        A solver that evaluates constraints itself, rather than through
        :meth:`~edelweissfe.solvers.base.nonlinearsolverbase.NonlinearSolverBase.assembleConstraintForces`, calls this
        whenever the constraint is located anew: the default :meth:`computeKernelsExplicit` does not size its tangent
        again on its own, and a block whose size has changed (a contact with other slave nodes) would not fit it.
        """

        self._discardedTangentScratch = None

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
        analysis; the scratch is sized once per discrete system: a solver that evaluates constraints itself calls
        :meth:`prepareExplicitEvaluation` whenever the constraint is located anew.
        A constraint whose tangent is costly overrides it, e.g. through
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

        # The discarded tangent buffer is sized once per discrete system, and re-zeroed in every increment.
        scratch = self._discardedTangentScratch
        if scratch is None:
            scratch = self._discardedTangentScratch = np.zeros(countEntriesOfBlock(self))
        else:
            scratch[:] = 0.0

        K = self.shapeBlock(scratch)
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
        whose sign alone is wrong. A ``P`` and a ``K`` negated together pass that check, so a test
        checks the direction of the forces the constraint exerts as well.

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
            The tangent ``dP/dU`` to augment, in the shape :meth:`shapeBlock` gives it (a
            dense ``nDof x nDof`` view unless the constraint declares a sparser pattern). A
            constraint that inherits
            :class:`~edelweissfe.constraints.base.forcesonlyexplicitevaluation.ForcesOnlyExplicitEvaluation`
            is also called with ``K=None`` and must then skip the tangent.
        timeStep
            The current time step.
        """
