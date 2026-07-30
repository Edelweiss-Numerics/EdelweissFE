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

from edelweissfe.steps.base.stepbase import StepBase, StepIncrementationSchema
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper


class AdaptiveStep(StepBase):
    """
    An adaptive incremental step to be used in nonlinear simulations with implicit time integration.
    """

    #: L2 schema declared for the L3 registry, per OptionSchemaProvider. Shared with
    #: :class:`~edelweissfe.steps.adaptivestepforexplicitsimulations.AdaptiveStepForExplicitSimulations`
    #: -- every step type accepts exactly the same incrementation options (see
    #: :meth:`~edelweissfe.steps.base.stepbase.StepBase.__init__`, which validates/coerces against
    #: it directly; no further construction happens here).
    schema = StepIncrementationSchema

    def _createTimeStepper(self) -> AdaptiveTimeStepper:
        return AdaptiveTimeStepper(
            self.model.time,
            self.length,
            self.startIncrementSize,
            self.maxIncrementSize,
            self.minIncrementSize,
            self.maxNumberIncrements,
            self.journal,
        )
