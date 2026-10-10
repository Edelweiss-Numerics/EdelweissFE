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

The datalines are joined line by line and executed as Python source, with the model tree bound
to the name ``model``.

Leading whitespace of datalines is not preserved by the input file parser, so indented blocks
(loops, conditionals, function bodies) cannot be written with plain spaces or tabs. Instead,
indent with the literal two-character sequence ``\\t``: every occurrence is replaced by a tab
character before the code is executed, so ``\\t\\t`` indents a line by two levels.

.. code-block:: edelweiss
    :caption: Indented Python code in a generator. Example:

    *modelGenerator, generator=executePythonCode, name=gen-2
    from edelweissfe.sets.nodeset import NodeSet
    load_nodes = []
    for n in model.nodes.values():
    \\tif abs( n.coordinates[0] - 495 ) <= 20:
    \\t\\tif abs( n.coordinates[1] - 81 ) <= 1e-12:
    \\t\\t\\tload_nodes.append(n)
    model.nodeSets['load'] = NodeSet('load', load_nodes)

Note that ``\\t`` is replaced everywhere in the code, including inside string literals.
"""

from edelweissfe.generators.base.generatorbase import GeneratorBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel


class Generator(GeneratorBase):
    """Directly execute Python code to create the model tree."""

    #: This generator's datalines are raw code, not a flat option mapping -- there is nothing to
    #: validate/coerce against a schema, so it declares none and overrides
    #: :meth:`fromGeneratorDefinition` instead of relying on the default implementation.
    schema = None

    def __init__(self, name: str, model: FEModel, journal: Journal, *, codeLines: str = ""):
        """Constructible standalone, with no parser involvement.
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
            Python source executed with ``model`` bound in its global namespace. Literal ``\\t``
            sequences are replaced by tab characters before execution, to allow indented blocks.
        """
        cleanCodeLines = codeLines.replace(r"\t", "\t")  # literal \t sequences become tabs, used for indentation

        exec(cleanCodeLines, {"model": model})

    @classmethod
    def fromGeneratorDefinition(cls, name: str, model: FEModel, journal: Journal, args: list, kwargs: dict) -> FEModel:
        """Build this generator from a parsed ``*modelGenerator`` definition.

        Overridden because this generator's datalines are raw Python source, not a flat
        ``key=value`` option mapping -- ``args`` (the parser's non-comma-split dataline strings,
        see the ``executePythoncode`` special case in ``helpers/inputfilehelpers.py``) is what
        carries them, not ``kwargs``."""
        cls(name, model, journal, codeLines="\n".join(args))
        return model
