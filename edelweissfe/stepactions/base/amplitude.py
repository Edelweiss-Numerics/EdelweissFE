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

"""Amplitude functions for step actions, and compilation of an input file's ``f(t)`` option into
one.

A step action prescribes a value that is scaled over the step by an amplitude. In Python that
amplitude *is* a callable -- which is what a step action's typed constructor takes. The ``f(t)``
option of an ``.inp`` file is a *serialization* of such a callable as a sympy expression string,
and turning it back into one is the job of each step action's ``fromStepActionDefinition`` /
``updateStepActionFromDefinition`` (see
:class:`~edelweissfe.stepactions.base.stepactionbase.StepActionBase`).

Centralized here so that every step action offering ``f(t)`` agrees on what an absent option means
-- see :func:`amplitudeFromExpression`.
"""

from collections.abc import Callable

import sympy as sp


def linearAmplitude(stepProgress: float) -> float:
    """The default amplitude: the prescribed value is reached linearly over the step.

    This is what a step action applies when no amplitude was given, i.e. what ``f_t=None`` means to
    a typed constructor.

    Parameters
    ----------
    stepProgress
        The progress of the step, in ``[0...1]``.

    Returns
    -------
    float
        The scaling factor to apply to the prescribed value.
    """

    return stepProgress


def amplitudeFromExpression(expression: str | None) -> Callable[[float], float] | None:
    """Compile an input file's ``f(t)`` expression into an amplitude function.

    Parameters
    ----------
    expression
        A sympy-parsable expression in the variable ``t``, e.g. ``"t**2"``. May be None, which is
        how the parser reports an omitted ``f(t)`` option.

    Returns
    -------
    Callable[[float], float] | None
        The compiled amplitude, or None if no expression was given. None is passed on to the step
        action's constructor rather than being replaced by :func:`linearAmplitude` here, so that
        "the user wrote no amplitude" stays distinguishable from "the user wrote ``f(t)='t'``" for
        as long as possible -- and so that a step action for which a missing amplitude is *not*
        meaningful can reject it (as ``changematerialproperty`` does).
    """

    if expression is None:
        return None

    t = sp.symbols("t")
    return sp.lambdify(t, sp.sympify(expression), "numpy")
