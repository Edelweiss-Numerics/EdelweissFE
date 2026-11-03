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
# Created on Mon Jan 23 13:05:53 2017

# @author: Matthias Neuner
"""
Step actions define actions run during a simulation.
They are defined within a ``*step`` definition, line by line,
by specifying their ``name`` and a list of ``option=value``, for example

.. code-block:: edelweiss

    *step, solver=mySolver
    >>dirichlet, name=bottom,   nSet=bottom,   field=displacement, 2=0, 1=0
    >>dirichlet, name=rightTop, nSet=rightTop, field=displacement, 2=0,
    >>distributedload, name=dload1, surface=top,  type=pressure, magnitude=10, f(t)=t*2
    >>distributedload, name=dload2, surface=left, type=pressure, magnitude=50, f(t)=t
"""

from edelweissfe.config import registry


def stepActionFactory(name: str) -> type:
    """Return the class implementing step action ``name``.

    Resolved through the registry (``stepaction`` category), which lets built-in step actions,
    entry points, and in-process :func:`~edelweissfe.config.registry.register` calls all be
    reached the same way.

    Parameters
    ----------
    name
        The name of the step action to load.

    Returns
    -------
    type
        The step action class type.

    Raises
    ------
    edelweissfe.config.registry.RegistryLookupError
        If no step action is registered under ``name``.
    """

    stepActionClass, _ = registry.lookup("stepaction", name)

    return stepActionClass
