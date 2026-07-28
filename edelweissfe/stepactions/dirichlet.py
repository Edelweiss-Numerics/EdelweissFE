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
# Created on Mon Jan 23 13:03:09 2017

# @author: Matthias Neuner

from collections.abc import Callable

import numpy as np
import sympy as sp

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Standard Dirichlet boundary condition.
If not modified in subsequent steps, the BC is held constant.
"""

inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []


def _addPrescriptionArgs(kw):
    """Register the value prescription arguments, which are shared by the
    'dirichlet' and 'updateDirichlet' keywords."""

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
    kw.addOptionalArg("analyticalField", "Scales the defined boundary condition", str, None)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)


for module in modules:
    kw = module.addOptionalKeyword("dirichlet", "Standard Dirichlet boundary condition.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("nSet", "The node set for application of the boundary condition.", str)
    kw.addRequiredArg("field", "Field for which the boundary condition is active.", str)
    _addPrescriptionArgs(kw)

    documentation.append(kw)

    kw = module.addOptionalKeyword("updateDirichlet", "Update a previously defined dirichlet definition.")
    kw.addRequiredArg("name", "Name of the step action to update.", str)
    _addPrescriptionArgs(kw)

    documentation.append(kw)


class StepAction(DirichletBase):
    """Dirichlet boundary condition, based on a node set.

    The constructor is typed: it takes the node set itself, the prescribed values as a mapping, and
    the amplitude as a callable. Nothing here parses an input file -- turning ``nSet=bottom``,
    ``2=0.5`` and ``f(t)='t**2'`` into those arguments is the job of
    :meth:`fromStepActionDefinition` below, which is the only part of this module the ``.inp``
    front-end needs. That split is what lets an external caller (EdelweissMeshfree, a script) use
    this class directly; the signature deliberately matches EdelweissMeshfree's own
    ``stepactions/dirichlet.py``, which exists only because this one could not be constructed
    without a parser-shaped dict.

    Parameters
    ----------
    name
        The name of this step action.
    nSet
        The node set the boundary condition is applied to.
    field
        The field the boundary condition acts on, e.g. ``"displacement"``.
    prescribedComponents
        Maps the zero-based component index of ``field`` to the value prescribed for it. Components
        absent from the mapping are left free.
    model
        The model tree.
    journal
        The journal object for logging.
    f_t
        The amplitude over the step progress interval ``[0...1]``. Defaults to the identity, i.e. the
        prescribed values are reached linearly at the end of the step.
    analyticalField
        Scales the prescribed values per node, evaluated at each node's coordinates.
    """

    def __init__(
        self,
        name,
        nSet,
        field,
        prescribedComponents: dict,
        model,
        journal,
        f_t: Callable[[float], float] = None,
        analyticalField=None,
    ):
        self.name = name

        self.field = field
        self.nSet = nSet
        self.fieldSize = getFieldSize(self.field, model.domainSize)

        self._components = None
        self._journal = journal
        self._model = model

        self.updateStepAction(prescribedComponents, f_t, analyticalField)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this boundary condition from a parsed ``>>dirichlet`` definition. See
        :class:`StepActionBase` for why this is separate from ``__init__``."""

        field = definition["field"]

        return cls(
            name,
            model.nodeSets[definition["nSet"]],
            field,
            cls._prescribedComponentsFromDefinition(definition, getFieldSize(field, model.domainSize)),
            model,
            journal,
            f_t=cls._amplitudeFromDefinition(definition),
            analyticalField=cls._analyticalFieldFromDefinition(definition, model),
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>dirichlet`` definition re-declared in a later step."""

        self.updateStepAction(
            self._prescribedComponentsFromDefinition(definition, self.fieldSize),
            self._amplitudeFromDefinition(definition),
            self._analyticalFieldFromDefinition(definition, model),
        )

    def _reconcileIfSetChanged(self):
        """Re-size the prescribed values if the node set was mutated in-place (e.g. AMR adding new
        boundary nodes) since the last check. Preserve the active/inactive flag: updateStepAction
        unconditionally activates, but a BC deactivated at a prior step end must stay inactive --
        otherwise a refinement in a later step would silently revive it. The node set itself needs
        no re-fetch: it has stable identity (mutated in place), so ``self.nSet`` is already current.

        Replays the *typed* state rather than a stashed definition dict, which is why
        ``__init__`` keeps ``prescribedComponents``/``f_t``/``analyticalField`` as attributes: the
        replay used to depend on the dict having been mutated in place by the ``components=``
        handling, an interaction that only worked because both lived in the same dict."""
        if self._checkSetChanged(self.nSet):
            wasActive = self.active
            self.updateStepAction(self._prescribedComponents, self._f_t, self._analyticalField)
            self.active = wasActive

    @property
    def components(
        self,
    ):
        return self._components

    def applyAtStepEnd(self, model):
        self.active = False

    def updateStepAction(
        self,
        prescribedComponents: dict,
        f_t: Callable[[float], float] = None,
        analyticalField=None,
    ):
        """Prescribe a new set of values, amplitude and analytical field on the same node set.

        Parameters
        ----------
        prescribedComponents
            Maps the zero-based component index of the field to the value prescribed for it.
        f_t
            The amplitude over the step progress interval ``[0...1]``; the identity if omitted.
        analyticalField
            Scales the prescribed values per node.
        """
        self.active = True

        self._checkSetChanged(self.nSet)

        outOfRange = [index for index in prescribedComponents if not 0 <= index < self.fieldSize]
        if outOfRange:
            raise ValueError(
                f"Dirichlet '{self.name}': component index/indices {sorted(outOfRange)} do not exist on "
                f"field '{self.field}', which has {self.fieldSize} component(s)."
            )

        self._prescribedComponents = dict(prescribedComponents)
        self._f_t = f_t
        self._analyticalField = analyticalField

        self._components = sorted(self._prescribedComponents)

        self.delta = np.tile([self._prescribedComponents[i] for i in self._components], (len(self.nSet), 1))

        if analyticalField is not None:
            self.analyticalField = analyticalField
            for i, node in enumerate(self.nSet):
                self.delta[i, :] *= analyticalField.evaluateAtCoordinates(node.coordinates)[0][0]

        self.amplitude = f_t if f_t is not None else lambda x: x

    def getPrescribedIncrement(self, timeStep: TimeStep):
        self._reconcileIfSetChanged()
        if self.active:
            return self.delta * (
                self.amplitude(timeStep.stepProgress)
                - (self.amplitude(timeStep.stepProgress - timeStep.stepProgressIncrement))
            )
        else:
            return self.delta * 0.0

    @staticmethod
    def _prescribedComponentsFromDefinition(definition: dict, fieldSize: int) -> dict:
        """Collect the prescribed values from a parsed definition's ``1``..``6`` and ``components``.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.
        fieldSize
            The number of components the field has.

        Returns
        -------
        dict
            Maps the zero-based component index to its prescribed value.
        """

        prescribed = {
            index: float(definition[str(index + 1)])
            for index in range(fieldSize)
            if definition[str(index + 1)] is not None
        }

        if definition["components"] is not None:
            # An entry of `x` marks a component as free; anything else overrides a numbered option
            # for the same component, which is the precedence the in-place dict mutation this
            # replaces happened to produce.
            values = np.array(eval(definition["components"].replace("x", "np.nan")), dtype=float)
            prescribed.update({index: value for index, value in enumerate(values) if not np.isnan(value)})

        return prescribed

    @staticmethod
    def _amplitudeFromDefinition(definition: dict) -> Callable[[float], float]:
        """Compile a parsed definition's ``f(t)`` expression into an amplitude function.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.

        Returns
        -------
        Callable[[float], float]
            The amplitude as a function of step progress, or None if none was specified.
        """

        if definition["f(t)"] is None:
            return None

        t = sp.symbols("t")
        return sp.lambdify(t, sp.sympify(definition["f(t)"]), "numpy")

    @staticmethod
    def _analyticalFieldFromDefinition(definition: dict, model):
        """Resolve a parsed definition's ``analyticalField`` name against the model.

        Parameters
        ----------
        definition
            The parsed option mapping defining this step action.
        model
            The model tree.

        Returns
        -------
        The analytical field, or None if none was specified.
        """

        if definition["analyticalField"] is None:
            return None

        return model.analyticalFields[definition["analyticalField"]]
