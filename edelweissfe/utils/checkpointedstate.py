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
"""The state a component carries from one increment to the next, written to and read from a restart
checkpoint as it is.

A component *declares* that state: a mapping of attribute name to type, for instance the implicit
solver's

.. code-block:: python

    checkpointedState = {"prevTimeStep": TimeStep, "dU": np.ndarray}

and :func:`packState` / :func:`unpackState` turn the declared attributes into the flat mapping of
arrays a checkpoint stores, and back. An empty mapping declares that a component carries nothing
between increments. A component whose declaration is still None has not been declared at all, and
cannot be checkpointed.

An attribute that is None is simply left out, and read back as None. A ``dict`` maps strings to
floats. An attribute whose type is none of the plain ones is a helper object that declares its own
``checkpointedState``; its state is stored under the attribute's name as a prefix.
"""

import numpy as np

from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import RestartError

#: How each type a declared attribute may have is stored ...
_TO_ARRAY = {
    float: lambda value: np.array(value, dtype=float),
    int: lambda value: np.array(value, dtype=np.int64),
    bool: lambda value: np.array(value, dtype=bool),
    np.ndarray: np.asarray,
    list: np.asarray,
    set: lambda value: np.array(sorted(value), dtype=bytes),
    TimeStep: lambda t: np.array(
        [t.number, t.stepProgressIncrement, t.stepProgress, t.timeIncrement, t.stepTime, t.totalTime]
    ),
}
#: ... and read back.
_FROM_ARRAY = {
    float: float,
    int: int,
    bool: bool,
    np.ndarray: np.array,
    list: list,
    set: lambda a: {entry.decode() for entry in a},
    TimeStep: lambda a: TimeStep(int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5])),
}


def _declaration(component) -> dict:
    declaration = component.checkpointedState
    if declaration is None:
        raise RestartError(
            "{:} does not declare the state it carries between increments (checkpointedState), so it "
            "cannot be checkpointed".format(type(component).__name__)
        )
    return declaration


def packState(component) -> dict[str, np.ndarray]:
    """The declared state of ``component``, as a flat mapping of arrays.

    Parameters
    ----------
    component
        Any object with a ``checkpointedState`` declaration.

    Returns
    -------
    dict[str, numpy.ndarray]
        One array per declared attribute that is not None.
    """

    state = {}
    for name, kind in _declaration(component).items():
        value = component.__dict__[name]
        if value is None:
            continue
        if kind is dict:
            state[name + ".keys"] = np.array(list(value.keys()), dtype=bytes)
            state[name + ".values"] = np.array(list(value.values()), dtype=float)
        elif kind in _TO_ARRAY:
            state[name] = _TO_ARRAY[kind](value)
        else:
            state |= {name + "." + key: array for key, array in packState(value).items()}
    return state


def unpackState(component, state: dict[str, np.ndarray]):
    """Set the declared attributes of ``component`` from what :func:`packState` returned.

    Parameters
    ----------
    component
        Any object with a ``checkpointedState`` declaration.
    state
        The mapping of arrays.
    """

    for name, kind in _declaration(component).items():
        if kind is dict:
            keys = state.get(name + ".keys")
            component.__dict__[name] = (
                None if keys is None else {k.decode(): float(v) for k, v in zip(keys, state[name + ".values"])}
            )
        elif kind in _FROM_ARRAY:
            component.__dict__[name] = _FROM_ARRAY[kind](state[name]) if name in state else None
        else:
            prefix = name + "."
            nested = {key[len(prefix) :]: array for key, array in state.items() if key.startswith(prefix)}
            if not nested:
                component.__dict__[name] = None
                continue
            helper = component.__dict__.get(name)
            if helper is None:
                helper = kind.__new__(kind)
            unpackState(helper, nested)
            component.__dict__[name] = helper
