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
# Created on Tue Dec 17 08:26:01 2019

# @author: Matthias Neuner

from dataclasses import dataclass

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.math import createMathExpression
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import schemaField

"""
A simple integrator to compute the fracture energy by integrating a load-displacement curve.

.. code-block:: edelweiss
    :caption: Example:

    *output, type=fractureenergyintegrator, jobName=myJob, name=gfi
        forceFieldOutput=RF, displacementFieldOutput=U, fractureArea='100.0*1.0'
"""


@dataclass(frozen=True)
class FractureEnergyIntegratorSchema:
    """The options this output manager accepts, owned by this module and never mutated from
    outside it.
    """

    # `forceFieldOutput`/`displacementFieldOutput` are declared `required=True` explicitly, but are
    # still given `default=None` so that `FractureEnergyIntegratorSchema()` remains constructible
    # without arguments; `buildSchemaFromOptions` still enforces that an `.inp` file supplies them.
    forceFieldOutput: str | None = schemaField(
        description="fieldOutput for force (with time history).", dtype=str, default=None, required=True
    )
    displacementFieldOutput: str | None = schemaField(
        description="fieldOutput for displacement (with time history).", dtype=str, default=None, required=True
    )
    #: Field name ``f_x`` because ``f(x)`` is not a valid Python identifier; the input-file-facing
    #: option name is restored via ``optionName``.
    f_x: str = schemaField(
        description="Apply a model accessible function on the result.",
        dtype=str,
        default="1",
        optionName="f(x)",
    )


class OutputManager(OutputManagerBase):
    """Simple Integrator for fracture energy"""

    identification = "FEI"
    printTemplate = "{:}, {:}: {:}"

    #: Option schema for this output manager, per OptionSchemaProvider.
    schema = FractureEnergyIntegratorSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
        *,
        configuration: FractureEnergyIntegratorSchema = FractureEnergyIntegratorSchema(),
    ):
        """Constructible standalone, with no parser involvement. Options arrive as an
        already-validated, already-typed schema instance.

        Parameters
        ----------
        name
            The name of this output manager.
        model
            The model tree.
        fieldOutputController
            The field output controller instance.
        journal
            The journal instance for logging.
        plotter
            The plotter instance.
        configuration
            The options this output manager accepts; defaults to all-defaults.
        """
        self.name = name
        self.journal = journal
        self.monitorJobs = []
        self.fieldOutputController = fieldOutputController

        fractureArea = configuration.f_x
        # Legacy behaviour: any *falsy* value (in particular an explicitly empty string) falls
        # back to "x", not just an omitted option. A schema default of "x" alone would not
        # reproduce this, since an explicitly-empty option would otherwise stay empty.
        if not fractureArea:
            fractureArea = "x"

        self.fpF = self.fieldOutputController.fieldOutputs[configuration.forceFieldOutput]
        self.fpU = self.fieldOutputController.fieldOutputs[configuration.displacementFieldOutput]
        self.A = createMathExpression(fractureArea)(0.0)
        self.fractureEnergy = 0.0

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        pass

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(
        self,
    ):
        pass

    def finalizeJob(
        self,
    ):
        self.fractureEnergy = np.trapezoid(self.fpF.getResultHistory(), x=self.fpU.getResultHistory()) / self.A
        self.journal.message(
            "integrated fracture energy: {:3.4f}".format(self.fractureEnergy),
            self.identification,
        )
