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
#  Matthias Neuner Matthias.Neuner@uibk.ac.at
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
# Created on 2022-09-08

# @author: Matthias Neuner
"""
Directly execute Python code to create the model tree.
"""

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.inputlanguage import InputLanguage, Module

module = Module("executePythoncode", "Directly execute Python code to create the model tree.")

inputLanguage = InputLanguage()

keyword = "modelGenerator"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredDatalines("Python code to run", str)

documentation = [module]


class Generator(GeneratorBase):
    """Directly execute Python code to create the model tree."""

    #: This generator's datalines are raw code, not a flat option mapping -- there is nothing to
    #: validate/coerce against a schema, so it declares none and overrides
    #: :meth:`fromGeneratorDefinition` instead of relying on the default implementation.
    schema = None

    def __init__(self, name: str, model: FEModel, journal: Journal, *, codeLines: str = ""):
        """L1: constructible standalone, with no ``InputLanguage``/``Module``/parser involvement.
        Populates ``model`` directly; construction *is* the generation.

        Parameters
        ----------
        name
            Unused: this generator names no sets of its own.
        model
            The model tree to populate. Mutated in place.
        journal
            Unused.
        codeLines
            Python source executed with ``model`` bound in its global namespace.
        """
        exec(codeLines, {"model": model})

    @classmethod
    def fromGeneratorDefinition(cls, name: str, model: FEModel, journal: Journal, args: list, kwargs: dict) -> FEModel:
        """Build this generator from a parsed ``*modelGenerator`` definition.

        Overridden because this generator's datalines are raw Python source, not a flat
        ``key=value`` option mapping -- ``args`` (the parser's non-comma-split dataline strings,
        see the ``executePythoncode`` special case in ``helpers/inputfilehelpers.py``) is what
        carries them, not ``kwargs``."""
        cls(name, model, journal, codeLines="\n".join(args))
        return model
