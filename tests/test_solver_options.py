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
#  ---------------------------------------------------------------------
"""Parsing of the ``*solver`` datalines into a solver's typed options.

All solvers share :meth:`NonlinearSolverBase._updateOptions`; these tests pin down the conversions
that a plain ``type(default)(value)`` gets wrong.
"""

import pytest

from edelweissfe.journal.journal import Journal
from edelweissfe.solvers.nonlinearexplicitdynamic import NED


def _ned(**options):
    return NED(jobInfo={}, journal=Journal(verbose=False), **options)


def test_boolean_option_false_is_parsed_as_false():
    # bool("False") is truthy; the option must be parsed, not cast.
    assert _ned(**{"report-performance": "False"}).options["report-performance"] is False
    assert _ned(**{"report-performance": "True"}).options["report-performance"] is True


def test_list_option_is_split_at_commas():
    solver = _ned(**{"expect-second-order-fields": "displacement, rotation"})
    assert solver.options["expect-second-order-fields"] == ["displacement", "rotation"]


def test_list_option_default_is_not_shared_between_solvers():
    _ned(**{"expect-first-order-fields": "temperature"})
    assert _ned().options["expect-first-order-fields"] == []


def test_unknown_option_raises():
    with pytest.raises(AttributeError):
        _ned(**{"courant-numbr": "0.5"})
