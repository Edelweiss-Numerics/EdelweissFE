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
from edelweissfe.utils.schema import OptionSchemaProvider
from edelweissfe.variables.scalarvariable import ScalarVariable


class ConstraintBase(OptionSchemaProvider, ABC, VIJEntityBase):
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

        If scalar variables are used, EdelweissFE expects the layout of the external load vector PExt
        (and the stiffness) to be of the form

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

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """Return this constraint's converged internal history (state not covered by its
        :attr:`scalarVariables`, e.g. frictional-contact history) to be serialized by
        :meth:`~edelweissfe.models.femodel.FEModel.writeRestart`, or ``None`` if the constraint is
        stateless.

        The default implementation returns ``None``, which is correct for every constraint that
        does not override it (its full state is either recomputed each increment or already covered
        by :attr:`scalarVariables`).

        Returns
        -------
        dict[str, np.ndarray] | None
            A flat mapping of array name to array, or ``None``.
        """

        return None

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore this constraint's converged internal history from a restart checkpoint.

        Parameters
        ----------
        data
            The mapping previously returned by :meth:`getRestartData`.
        """

        raise NotImplementedError("This constraint does not carry restartable internal history.")

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

    def _checkSetChanged(self, theSet) -> bool:
        """Lazily detect whether ``theSet`` (a stable-identity
        :class:`~edelweissfe.sets.nodeset.NodeSet` or :class:`~edelweissfe.sets.elementset.ElementSet`)
        was mutated in-place (e.g. by AMR) since this constraint last checked it.

        A constraint that pre-sizes a derived array to the set's size calls this at its own
        per-increment entry point (e.g. :meth:`updateConnectivity`) to recompute that array
        lazily, without registering as a
        :class:`~edelweissfe.models.meshdependent.MeshDependent`.

        Parameters
        ----------
        theSet
            The set whose version is being tracked.

        Returns
        -------
        bool
            True once per version bump of ``theSet`` since the last call for this same set.
        """
        setVersions = self.__dict__.setdefault("_setVersions", {})
        key = id(theSet)
        changed = setVersions.get(key, theSet._version) != theSet._version
        setVersions[key] = theSet._version
        return changed

    def applyConstraintForcesOnly(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Evaluate this constraint's nodal forces without producing a tangent.

        What an explicit solver needs: it assembles no system matrix, so any tangent a constraint
        computes is built and discarded. The default implementation below still asks for one, through
        the same ``getVIJContributionSize``/``shapeVIJContribution`` protocol the implicit system
        matrix uses, so every existing constraint keeps working without change.

        A constraint for which the tangent is a material part of the cost should override this.
        ``nodeToDeformableSurfacePenalty`` does: its per-slave outer product and four block writes are
        2.5 ms per increment on a 280k-dof contact model, and its container another 1.2 ms, all of it
        thrown away.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The current solution increment, likewise restricted.
        PExt
            The local force vector to augment.
        timeStep
            The current time step.
        """

        K = self.shapeVIJContribution(np.zeros(self.getVIJContributionSize()))
        self.applyConstraint(U_np, dU, PExt, K, timeStep)

    @abstractmethod
    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        V: np.ndarray,
        timeStep: TimeStep,
    ):
        """Apply the constraint.  Add the contributions to the external load vector and the system matrix.

        Parameters
        ----------
        U_np
            The current total solution vector.
        dU
            The current increment since the last time the constraint was applied.
        PExt
            The external load vector.
        K
            The system (stiffness) matrix.
        dT
            The time increment.
        time
            The current step and total time.
        """
