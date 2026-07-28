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
# Created on Tue Jan 24 19:33:06 2017

# @author: Matthias Neuner

from collections.abc import Callable

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.amplitude import (
    amplitudeFromExpression,
    linearAmplitude,
)
from edelweissfe.stepactions.base.nodalloadbase import NodalLoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Apply node forces on a nSet.
"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("nodeforces", "Apply node forces on node sets.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("nSet", "The node set for application of the boundary condition.", str)
    kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)

    kw.addOptionalArg("1", "Prescribe first component of field.", float, None)
    kw.addOptionalArg("2", "Prescribe second component of field.", float, None)
    kw.addOptionalArg("3", "Prescribe third component of field.", float, None)
    kw.addOptionalArg("4", "Prescribe fourth component of field.", float, None)
    kw.addOptionalArg("5", "Prescribe fifth component of field.", float, None)
    kw.addOptionalArg("6", "Prescribe sixth component of field.", float, None)

    kw.addOptionalArg(
        "components",
        "Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        str,
        None,
    )
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)

    kw = module.addOptionalKeyword("updateNodeforces", "Update a previously defined nodeforces definition.")
    kw.addRequiredArg("name", "Name of the step action to update.", str)
    # kw.addRequiredArg("nSet", "The node set for application of the boundary condition.", str)
    # kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)

    kw.addOptionalArg("1", "Prescribe first component of field.", float, None)
    kw.addOptionalArg("2", "Prescribe second component of field.", float, None)
    kw.addOptionalArg("3", "Prescribe third component of field.", float, None)
    kw.addOptionalArg("4", "Prescribe fourth component of field.", float, None)
    kw.addOptionalArg("5", "Prescribe fifth component of field.", float, None)
    kw.addOptionalArg("6", "Prescribe sixth component of field.", float, None)

    kw.addOptionalArg(
        "components",
        "Prescribe values using a numpy ndarray for representation; use 'x' for ignored values.",
        str,
        None,
    )
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)


class StepAction(NodalLoadBase):
    """Defines node based load, defined on a nodeset.

    The constructor is typed: it takes the node set itself, the per-node force vector as an
    ``np.ndarray`` and the amplitude as a callable. Nothing here parses an input file -- turning
    ``nSet=gen_top``, ``2=-50`` (or ``components='0,-1000'``) and ``f(t)='sin(t*2*pi)'`` into those
    arguments is the job of :meth:`fromStepActionDefinition` below, which is the only part of this
    module the ``.inp`` front-end needs.

    The load accumulates *over steps*: the force reached at the end of a step is remembered, and what
    a later step declares is applied on top of it. See :meth:`updateStepAction`.

    Parameters
    ----------
    name
        The name of this step action.
    nSet
        The node set the forces are applied to.
    field
        The field the forces act on, e.g. ``"displacement"``.
    nodeForces
        The force applied to every node of the set, with one entry per component of ``field``.
    model
        The model tree.
    journal
        The journal object for logging.
    f_t
        The amplitude over the step progress interval ``[0...1]``. Defaults to the identity, i.e. the
        forces are reached linearly at the end of the step.
    """

    def __init__(
        self,
        name,
        nSet,
        field,
        nodeForces: np.ndarray,
        model,
        journal,
        f_t: Callable[[float], float] = None,
    ):
        self.name = name

        self._field = field
        self._nSet = nSet

        self._journal = journal
        self._model = model

        self._fieldSize = getFieldSize(self._field, model.domainSize)

        shape = (len(self._nSet), self._fieldSize)

        self.nodeForcesStepStart = np.zeros(shape)
        self.nodeForcesDelta = np.zeros(shape)
        self._nSetNodeOrder = list(self._nSet)  # node identity per row, for the lazy resize below

        self.updateStepAction(nodeForces, f_t=f_t)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build these node forces from a parsed ``>>nodeforces`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``."""

        field = definition["field"]

        return cls(
            name,
            model.nodeSets[definition["nSet"]],
            field,
            cls._nodeForcesFromDefinition(definition, getFieldSize(field, model.domainSize)),
            model,
            journal,
            f_t=amplitudeFromExpression(definition["f(t)"]),
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>nodeforces`` definition re-declared in a later step.

        Neither ``nSet`` nor ``field`` may be read here: a *partial* re-declaration
        (``>>nodeforces, name=cloadTop, 2=-10, f(t)=sin(t*2*pi)``, as in
        ``testfiles/marmot/NodeForces/test.inp``) omits them, and the parser then validates the line
        against the ``update`` + keyword-name schema, i.e. the ``updateNodeforces`` keyword declared
        above (``utils/inputfileparser.py``, ``parseModuleKeywordLine``), whose definition contains
        no ``nSet``/``field`` keys at all. The field size therefore comes from the instance -- which
        is also the right source, since the node set a load acts on cannot change. Note that only
        the *schema* is live: a ``>>updateNodeforces`` keyword *line* is unroutable, because the
        parser would look for a step action module of that name (see PLAN_INPUT_SYSTEM.md's P3
        row)."""

        self.updateStepAction(
            self._nodeForcesFromDefinition(definition, self._fieldSize),
            f_t=amplitudeFromExpression(definition["f(t)"]),
        )

    def _reconcileIfSetChanged(self):
        """Re-size the load arrays if the node set was mutated in-place (e.g. AMR adding new
        boundary nodes) since the last check, preserving each retained node's accumulated/pending
        load by identity; newly added nodes get zero force. Without this, the flat load array
        would no longer match the node set's DOF layout after refinement grows a loaded boundary.
        The node set itself needs no re-fetch: it has stable identity (mutated in place), so
        ``self._nSet`` is already current -- only these derived, pre-sized arrays go stale."""
        if not self._checkSetChanged(self._nSet):
            return
        oldStart = {node: self.nodeForcesStepStart[i] for i, node in enumerate(self._nSetNodeOrder)}
        oldDelta = {node: self.nodeForcesDelta[i] for i, node in enumerate(self._nSetNodeOrder)}
        newNodes = list(self._nSet)
        shape = (len(newNodes), self._fieldSize)
        self.nodeForcesStepStart = np.zeros(shape)
        self.nodeForcesDelta = np.zeros(shape)
        for i, node in enumerate(newNodes):
            if node in oldStart:
                self.nodeForcesStepStart[i] = oldStart[node]
                self.nodeForcesDelta[i] = oldDelta[node]
        self._nSetNodeOrder = newNodes

    def updateStepAction(self, nodeForces: np.ndarray, f_t: Callable[[float], float] = None):
        """Prescribe a new force vector and amplitude on the same node set.

        The load accumulated up to the start of this step stays untouched; ``nodeForces`` is the
        increment applied on top of it during this step -- unlike ``bodyforce``/``distributedload``,
        which additionally accept a new *total*.

        Parameters
        ----------
        nodeForces
            The force *increment* applied to every node of the set during this step, with one entry
            per component of the field.
        f_t
            The amplitude over the step progress interval ``[0...1]``; the identity if omitted.
        """

        self._reconcileIfSetChanged()
        self._idle = False

        if len(nodeForces) != self._fieldSize:
            raise ValueError(
                f"NodeForces '{self.name}': {len(nodeForces)} force component(s) given, but field "
                f"'{self._field}' has {self._fieldSize} component(s)."
            )

        self.nodeForcesDelta = np.tile(nodeForces, (len(self._nSet), 1))
        self.amplitude = f_t if f_t is not None else linearAmplitude

    @property
    def field(self) -> str:
        """The field these forces act on.

        Returns
        -------
        str
            The name of the field.
        """

        return self._field

    @property
    def nodeSet(self) -> NodeSet:
        """The nodes these forces are acting on.

        Returns
        -------
        NodeSet
            The node set.
        """

        return self._nSet

    @staticmethod
    def _nodeForcesFromDefinition(definition: dict, fieldSize: int) -> np.ndarray:
        """Collect the per-node force vector from a parsed definition's ``1``..``6`` and
        ``components``.

        Note the deliberate difference to ``dirichlet``, which offers options of the same names: an
        entry of ``x`` means *free* to a boundary condition, so dirichlet maps it to ``np.nan`` and
        uses the result as a mask, whereas an unloaded component simply carries no force, so here
        ``x`` means the value zero and the result is a dense vector. Do not unify the two.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.
        fieldSize
            The number of components the field has.

        Returns
        -------
        np.ndarray
            The force applied to every node of the set, one entry per field component.
        """

        if definition["components"] is not None:
            return np.asarray(eval(definition["components"].replace("x", "0")), dtype=float)

        nodeForces = np.zeros(fieldSize)

        for index in range(fieldSize):
            if definition[str(index + 1)] is not None:
                nodeForces[index] = float(definition[str(index + 1)])

        return nodeForces

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Fold this step's increment into the accumulated forces and go idle.

        Idle means "no increment pending": until a later step re-declares this load, it stays at the
        accumulated level.

        Parameters
        ----------
        model
            The current state of the model.
        stepMagnitude
            The fraction of the increment that was actually applied. None means the full increment,
            i.e. the amplitude evaluated at the end of the step; the arc length solvers pass their
            load parameter here instead.
        """

        self._reconcileIfSetChanged()
        if not self._idle:
            if stepMagnitude is None:
                # standard case
                self.nodeForcesStepStart += self.nodeForcesDelta * self.amplitude(1.0)
            else:
                # set the 'actual' increment manually, e.g. for arc length method
                self.nodeForcesStepStart += self.nodeForcesDelta * stepMagnitude

            self.nodeForcesDelta[:] = 0
            self._idle = True

    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """The nodal forces at the current point of the step.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The accumulated forces plus the amplitude-scaled increment of this step, one row per node
            of the set.
        """

        self._reconcileIfSetChanged()
        if self._idle:
            return self.nodeForcesStepStart
        else:
            t = timeStep.stepProgress
            amp = self.amplitude(t)

            return self.nodeForcesStepStart + self.nodeForcesDelta * amp
