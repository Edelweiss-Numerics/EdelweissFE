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

from abc import ABC, abstractmethod

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.schema import OptionSchemaProvider, buildSchemaFromOptions


class GeneratorBase(OptionSchemaProvider, ABC):
    """Base class for model generators.

    Unlike sections/constraints/output managers, a generator has no further interface beyond its
    constructor: populating (or otherwise mutating) ``model`` *is* what a generator does, and it
    does so directly in ``__init__`` -- there is no separate ``apply``/``solve`` method every
    generator must also implement.

    A generator is reached either from Python or from an ``.inp`` file, and the input file is a
    *serialization* of the Python path, not a second way of building the model contribution. A
    ported generator therefore declares a real typed ``__init__`` and an L2 :attr:`schema`; the
    default :meth:`fromGeneratorDefinition` below is the only translation the ``.inp`` front-end
    needs, for the common case (a flat set of scalar options, no structural name to resolve against
    the model -- a generator *creates* sets/elements, it does not reference existing ones by name).
    Override :meth:`fromGeneratorDefinition` for a generator whose datalines are not a flat option
    mapping (e.g. ``executePythonCode``'s raw code lines).
    """

    @classmethod
    def fromGeneratorDefinition(cls, name: str, model: FEModel, journal: Journal, args: list, kwargs: dict) -> FEModel:
        """Create this generator from a parsed ``*modelGenerator`` definition.

        This is the L4 seam: the one place a generator's input-file shape (string-typed datalines)
        is turned into the typed arguments its real constructor takes. The default implementation
        covers every generator whose datalines are a flat ``key=value`` option mapping, which is
        every one built into EdelweissFE except ``executePythonCode``.

        Parameters
        ----------
        name
            The name of this generator instance (from the ``*modelGenerator`` keyword's own
            required ``name`` argument), used by every built-in generator as the prefix for the
            node/element sets it creates.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        args
            The generator's positional (non-``key=value``) dataline tokens. Ignored by every
            built-in generator except ``executePythonCode``, which overrides this method to use
            them instead of ``kwargs``.
        kwargs
            The generator's ``key=value`` dataline options.

        Returns
        -------
        FEModel
            The (mutated) model tree.

        Raises
        ------
        ValueError
            If this generator declares no L2 schema -- the default implementation cannot translate
            ``kwargs`` into a typed constructor call without one.
        """
        if cls.schema is None:
            raise ValueError(
                f"Generator '{cls.__name__}' declares no option schema. Declare one as a `schema` "
                "class attribute (see `edelweissfe.utils.schema.OptionSchemaProvider`)."
            )

        configuration = buildSchemaFromOptions(cls.schema, kwargs)
        cls(name, model, journal, configuration=configuration)
        return model

    @abstractmethod
    def __init__(self, name: str, model: FEModel, journal: Journal, *, configuration=None):
        """The generator base constructor.

        Parameters
        ----------
        name
            The name of this generator instance, used as the prefix for generated sets.
        model
            The model tree to populate. Mutated in place.
        journal
            The journal object for logging.
        configuration
            The L2 schema instance carrying this generator's options.
        """
