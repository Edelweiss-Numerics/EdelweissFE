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

from collections.abc import Callable

from edelweissfe.stepactions.base.amplitude import amplitudeFromExpression
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage

"""
Stepaction to change material properties.

"""


inputLanguage = InputLanguage()

# Register this step action for all available step types. This requires the step type
# modules to be imported before the step actions, as done in the input file parser.
modules = inputLanguage["step"].modules if "step" in inputLanguage else []

documentation = []

for module in modules:
    kw = module.addOptionalKeyword("changematerialproperty", "Stepaction to change material properties.")
    kw.addRequiredArg("name", "Name of the step action.", str)
    kw.addRequiredArg("material", "The id of the material to be changed", str)
    kw.addRequiredArg("index", "The index of the property in the material properties vector", int)
    kw.addOptionalArg("f(t)", "Define an amplitude in the step progress interval [0...1]", str, None)

    documentation.append(kw)


class StepAction(StepActionBase):
    """Drive one entry of a material's property vector along a prescribed function of time.

    The constructor is typed: it takes the material itself and the property function as a callable.
    Resolving ``material=myMaterial`` against the model and compiling ``f(t)='1000 - 500 * t'`` into
    that callable is the job of :meth:`fromStepActionDefinition`, which is the only part of this
    module the ``.inp`` front-end needs.

    Parameters
    ----------
    name
        The name of this step action.
    material
        The material whose property is changed, as the model holds it: a
        ``{"name", "properties"}`` record for the Marmot material provider, or a material instance
        for the edelweiss material provider.
    index
        The zero-based index of the property in the material's property vector.
    f_t
        The value of the property as a function of time. Required and positional, unlike the optional
        ``f_t`` amplitude of the other step actions, because driving the property along it is this
        action's entire purpose -- there is no meaningful default.

        Note that ``f_t`` is evaluated at the **absolute step time**, not at the step progress in
        ``[0...1]`` as everywhere else: it yields the property value itself rather than a factor
        scaling one. That asymmetry is intentional, and is why the input file's ``f(t)`` for this
        keyword means something different than for e.g. ``>>dirichlet``.
    model
        The model tree. Accepted for uniformity with the other step actions; the sections carrying
        this material are looked up in the model handed to :meth:`applyAtIncrementStart`.
    journal
        The journal object for logging.
    """

    def __init__(self, name, material, index: int, f_t: Callable[[float], float], model, journal):
        self.name = name
        self.theMaterial = material
        self.theIndex = int(index)

        self._journal = journal

        self.updateStepAction(f_t)

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        """Build this step action from a parsed ``>>changeMaterialProperty`` definition. See
        :class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase` for why this is separate
        from ``__init__``.

        Parameters
        ----------
        name
            The name of the step action.
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.

        Returns
        -------
        StepAction
            The constructed step action.
        """

        f_t = amplitudeFromExpression(definition["f(t)"])

        if f_t is None:
            raise ValueError(
                f"changeMaterialProperty '{name}': option 'f(t)' is required, as it defines the value "
                "the material property is set to at a given time."
            )

        return cls(
            name,
            CaseInsensitiveDict(model.materials)[definition["material"]],
            int(definition["index"]),
            f_t,
            model,
            journal,
        )

    def updateStepActionFromDefinition(self, definition, jobInfo, model, fieldOutputController, journal):
        """Update from a parsed ``>>changeMaterialProperty`` definition re-declared in a later step.

        The re-declared ``material`` and ``index`` are ignored, as they always have been: only the
        property function can be replaced, and the action is armed again.

        Parameters
        ----------
        definition
            The parsed option mapping for this step action.
        jobInfo
            A dictionary containing the information about the job.
        model
            The model tree.
        fieldOutputController
            The field output controlling object.
        journal
            The journal object for logging.
        """

        self.updateStepAction(amplitudeFromExpression(definition["f(t)"]))

    def updateStepAction(self, f_t: Callable[[float], float] = None):
        """Prescribe a new property function, and set the action active again.

        Parameters
        ----------
        f_t
            The new value of the property as a function of the absolute step time. If None, the
            function prescribed so far is kept -- which is what a re-declaration of this step action
            without a ``f(t)`` option means. On construction it is required instead, since there is
            then no function to keep.
        """

        self.active = True

        if f_t is not None:
            self.f_t = f_t

    def applyAtStepEnd(self, model, stepMagnitude=None):
        """Deactivate this step action at the end of the step it was declared in.

        Parameters
        ----------
        model
            The current state of the model.
        stepMagnitude
            Unused; present for signature compatibility with the other step actions.
        """

        self.active = False

    def applyAtIncrementStart(self, model, timeStep: TimeStep):
        """Change the actual properties depending on the current step time.

        Parameters
        ----------
        model
            The current state of the model.
        timeStep
            The definition of the time increment.
        """

        if not self.active:
            return

        theCurrentProperty = self.f_t(timeStep.stepTime)
        self._journal.message(
            "Changing property[{:}] of material {:} to {:}".format(
                self.theIndex, self._materialLabel(), theCurrentProperty
            ),
            self.name,
        )

        modifiedProperties = self._propertyVector()
        modifiedProperties[self.theIndex] = theCurrentProperty

        for section in model.sections.values():
            if not self._sectionUsesThisMaterial(section):
                continue

            for elSet in section.elSets:
                for el in elSet:
                    # The material is rebuilt per element on purpose: an edelweiss material instance
                    # holds the state vars of the element it is assigned to, so one instance shared
                    # across an element set would alias them.
                    section.assignSectionPropertiesToElement(
                        el, material=self._materialWithProperties(section.material, modifiedProperties)
                    )

    def _materialLabel(self) -> str:
        """The name of the driven material, for logging.

        Returns
        -------
        str
            The Marmot material's name, or the class name of an edelweiss material instance, which
            carries no name of its own.
        """

        if isinstance(self.theMaterial, dict):
            return self.theMaterial["name"]

        return type(self.theMaterial).__name__

    def _propertyVector(self):
        """The property vector of the driven material, to be modified in place.

        Returns
        -------
        np.ndarray
            The properties of the Marmot material record, or of the edelweiss material instance.
        """

        if isinstance(self.theMaterial, dict):
            return self.theMaterial["properties"]

        return self.theMaterial.materialProperties

    def _sectionUsesThisMaterial(self, section) -> bool:
        """Whether a section's elements are to be handed the modified material.

        Parameters
        ----------
        section
            The section to test.

        Returns
        -------
        bool
            True if the section uses the driven material. A Marmot material is matched by name, an
            edelweiss material by identity, as an instance of it carries no name.
        """

        if isinstance(self.theMaterial, dict):
            return isinstance(section.material, dict) and section.material["name"] == self.theMaterial["name"]

        return section.material is self.theMaterial

    @staticmethod
    def _materialWithProperties(sectionMaterial, modifiedProperties):
        """Create a copy of a section's material carrying the modified property vector.

        Parameters
        ----------
        sectionMaterial
            The material currently assigned to the section.
        modifiedProperties
            The modified property vector.

        Returns
        -------
        The material to be assigned to the section's elements, of the same kind as
        ``sectionMaterial``.

        Notes
        -----
        The autodiff materials' energy density function needs no carrying over, although
        :mod:`edelweissfe.sections.base.sectionbase` carries it over at the equivalent point. It is
        a no-op there: ``_materialEnergy`` is assigned only by ``setEnergyFunction``, which in turn
        is called only from each material's own ``__init__`` with ``materialProperties["psi_e"]``
        (verified across both EdelweissFE and EdelweissMeshfree), so re-running ``__init__`` on a
        property set that still carries ``psi_e`` installs the very same function. The code that
        used to stand here was a copy of that site which named the attribute ``self.material``,
        which this class never defines -- so it raised ``AttributeError`` for *every* edelweiss
        material rather than only failing to do something unnecessary. ``sectionbase``'s copy is
        left alone; removing it is its own change.
        """

        if isinstance(sectionMaterial, dict):  # for marmotmaterial provider
            modifiedMaterial = sectionMaterial.copy()
            modifiedMaterial["properties"] = modifiedProperties
            return modifiedMaterial

        # for edelweissmaterial provider: rebuild the instance from the modified properties
        return type(sectionMaterial)(modifiedProperties)
