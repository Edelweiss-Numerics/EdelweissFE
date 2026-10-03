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
"""Every component that can be part of a checkpointed run states what it carries from one increment
to the next. A component that does not would be checkpointed as if it carried nothing, and a resumed
run would silently continue from the wrong state -- so a missing declaration fails here, for every
class in the package, whether or not a restart scenario happens to use it."""

import importlib
import inspect
import pkgutil

import pytest

import edelweissfe.constraints
import edelweissfe.outputmanagers
import edelweissfe.solvers
import edelweissfe.timesteppers
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.timesteppers.base.timestepperbase import TimeStepperBase

#: Solvers that do not support restart. Writing a checkpoint with one of them refuses.
_NOT_RESTARTABLE = {"NEST", "NESTParallel", "NISTPArcLength"}


def _concreteSubclasses(package, base) -> list[type]:
    classes = set()
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".test_" in module.name:
            continue
        try:
            imported = importlib.import_module(module.name)
        except ImportError:  # an optional compiled extension that is not built
            continue
        for _, cls in inspect.getmembers(imported, inspect.isclass):
            if issubclass(cls, base) and cls is not base and not inspect.isabstract(cls):
                classes.add(cls)
    return sorted(classes, key=lambda cls: cls.__module__ + "." + cls.__qualname__)


_COMPONENTS = [
    cls
    for package, base in [
        (edelweissfe.constraints, ConstraintBase),
        (edelweissfe.outputmanagers, OutputManagerBase),
        (edelweissfe.solvers, NonlinearSolverBase),
        (edelweissfe.timesteppers, TimeStepperBase),
    ]
    for cls in _concreteSubclasses(package, base)
]


@pytest.mark.parametrize("cls", _COMPONENTS, ids=lambda cls: cls.__module__ + "." + cls.__qualname__)
def test_every_component_declares_its_checkpointed_state(cls):
    declares = cls.checkpointedState is not None
    overrides = any(
        "getRestartData" in vars(klass)
        for klass in cls.__mro__[: cls.__mro__.index(object)]
        if klass not in (ConstraintBase, OutputManagerBase, NonlinearSolverBase, TimeStepperBase)
    )
    if cls.__name__ in _NOT_RESTARTABLE:
        assert not declares and not overrides, "{:} is listed as not restartable".format(cls.__name__)
        return
    assert declares or overrides, (
        "{:} declares no checkpointedState: say which attributes it carries from one increment to the "
        "next (an empty mapping if none), or override getRestartData/setRestartData".format(cls.__qualname__)
    )


def test_every_kind_of_component_was_found():
    """Guards the discovery above: an empty list would pass every check."""
    for base in (ConstraintBase, OutputManagerBase, NonlinearSolverBase, TimeStepperBase):
        assert any(issubclass(cls, base) for cls in _COMPONENTS), base.__name__
