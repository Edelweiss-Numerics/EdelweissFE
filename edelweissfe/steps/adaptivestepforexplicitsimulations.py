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
#  Alexander Dummer alexander.dummer@uibk.ac.at
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
from edelweissfe.timesteppers.simpletimestepper import SimpleTimeStepper


class AdaptiveStepForExplicitSimulations(StepBase):
    """
    An adaptive incremental step to be used in nonlinear simulations with explicit time integration.

    It always begins with the zero increment, in which the explicit solver builds its initial state before time
    advances (see :class:`~edelweissfe.timesteppers.simpletimestepper.SimpleTimeStepper`), and refuses
    ``makeZeroIncrementFirst=False``.
    """

    #: Option schema for this step, per OptionSchemaProvider. Shared with
    #: :class:`~edelweissfe.steps.adaptivestep.AdaptiveStep` -- see that class's own docstring.
    schema = StepIncrementationSchema

    def _createTimeStepper(self) -> SimpleTimeStepper:
        """The time stepper of the step, which begins with the zero increment.

        Returns
        -------
        SimpleTimeStepper
            The time stepper.

        Raises
        ------
        ValueError
            If the step is asked to begin without the zero increment (``makeZeroIncrementFirst=False``).
        """

        if not self.makeZeroIncrementFirst:
            raise ValueError(
                "step {:}: an explicit step always begins with the zero increment, in which the explicit solver builds "
                "its initial state; makeZeroIncrementFirst=False is an option of AdaptiveStep only".format(self.number)
            )

        return SimpleTimeStepper(
            self.model.time,
            self.length,
            self.startIncrementSize,
            self.maxIncrementSize,
            self.minIncrementSize,
            self.maxNumberIncrements,
            self.journal,
        )
